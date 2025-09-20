"""
Modular Trading Bot - Supports multiple exchanges
"""

import os
import time
import asyncio
import traceback
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
from datetime import datetime

from exchanges import ExchangeFactory
from helpers import TradingLogger
from helpers.lark_bot import LarkBot


@dataclass
class TradingConfig:
    """Configuration class for trading parameters."""
    ticker: str
    contract_id: str
    quantity: Decimal
    take_profit: Decimal
    tick_size: Decimal
    direction: str
    max_orders: int
    wait_time: int
    exchange: str
    grid_step: Decimal
    instance_id: str = "default"  # Unique identifier for the bot instance

    @property
    def close_order_side(self) -> str:
        """Get the close order side based on bot direction."""
        return 'buy' if self.direction == "sell" else 'sell'


@dataclass
class OrderMonitor:
    """Thread-safe order monitoring state."""
    order_id: Optional[str] = None
    filled: bool = False
    filled_price: Optional[Decimal] = None
    filled_qty: Decimal = 0.0

    def reset(self):
        """Reset the monitor state."""
        self.order_id = None
        self.filled = False
        self.filled_price = None
        self.filled_qty = 0.0


class TradingBot:
    """Modular Trading Bot - Main trading logic supporting multiple exchanges."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = TradingLogger(config.exchange, config.ticker, config.instance_id, log_to_console=True)

        # Create exchange client
        try:
            self.exchange_client = ExchangeFactory.create_exchange(
                config.exchange,
                config
            )
        except ValueError as e:
            raise ValueError(f"Failed to create exchange client: {e}")

        # Trading state
        self.active_close_orders = []
        self.last_close_orders = 0
        self.last_open_order_time = 0
        self.last_log_time = 0
        self.current_order_status = None
        self.order_filled_event = asyncio.Event()
        self.order_canceled_event = asyncio.Event()
        self.shutdown_requested = False
        self.loop = None

        # Enhanced hourly statistics
        self.hourly_operations = 0
        self.hourly_fees = Decimal(0)
        self.last_hour_reset = time.time()
        
        # Separate statistics for position operations and closing operations
        self.hourly_position_operations = 0  # Current Position related operations
        self.hourly_closing_operations = 0   # Active closing amount related operations
        self.hourly_successful_fills = 0     # Successfully filled orders
        self.hourly_canceled_orders = 0      # Canceled orders

        # Register order callback
        self._setup_websocket_handlers()

    async def send_startup_notification(self):
        """Send startup notification to Lark."""
        lark_token = os.getenv("LARK_TOKEN")
        if lark_token:
            try:
                async with LarkBot(lark_token) as bot:
                    startup_data = {
                        "quantity": str(self.config.quantity),
                        "take_profit": str(self.config.take_profit),
                        "direction": self.config.direction,
                        "max_orders": self.config.max_orders,
                        "wait_time": self.config.wait_time,
                        "grid_step": str(self.config.grid_step)
                    }
                    
                    await bot.send_notification(
                        notification_type="startup",
                        instance_id=self.config.instance_id,
                        exchange=self.config.exchange,
                        ticker=self.config.ticker,
                        message=f"Trading bot started with instance ID: {self.config.instance_id}",
                        data=startup_data
                    )
                    self.logger.log(f"Startup notification sent for instance: {self.config.instance_id}", "INFO")
            except Exception as e:
                self.logger.log(f"Failed to send startup notification: {e}", "ERROR")

    async def graceful_shutdown(self, reason: str = "Unknown"):
        """Perform graceful shutdown of the trading bot."""
        self.logger.log(f"Starting graceful shutdown: {reason}", "INFO")
        self.shutdown_requested = True

        try:
            # Disconnect from exchange
            await self.exchange_client.disconnect()
            self.logger.log("Graceful shutdown completed", "INFO")

        except Exception as e:
            self.logger.log(f"Error during graceful shutdown: {e}", "ERROR")

    def _setup_websocket_handlers(self):
        """Setup WebSocket handlers for order updates."""
        def order_update_handler(message):
            """Handle order updates from WebSocket."""
            try:
                # Check if this is for our contract
                if message.get('contract_id') != self.config.contract_id:
                    return

                order_id = message.get('order_id')
                status = message.get('status')
                side = message.get('side', '')
                order_type = message.get('order_type', '')
                filled_size = Decimal(message.get('filled_size'))
                if order_type == "OPEN":
                    self.current_order_status = status

                if status == 'FILLED':
                    if order_type == "OPEN":
                        # Ensure thread-safe interaction with asyncio event loop
                        if self.loop is not None:
                            self.loop.call_soon_threadsafe(self.order_filled_event.set)
                        else:
                            # Fallback (should not happen after run() starts)
                            self.order_filled_event.set()
                        
                        # Update position operation statistics
                        self.hourly_position_operations += 1
                        self.hourly_successful_fills += 1
                    elif order_type == "CLOSE":
                        # Update closing operation statistics
                        self.hourly_closing_operations += 1

                    self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                    f"{message.get('size')} @ {message.get('price')}", "INFO")
                    self.logger.log_transaction(order_id, side, message.get('size'), message.get('price'), status)
                elif status == "CANCELED":
                    if order_type == "OPEN":
                        self.order_filled_amount = filled_size
                        if self.loop is not None:
                            self.loop.call_soon_threadsafe(self.order_canceled_event.set)
                        else:
                            self.order_canceled_event.set()

                        # Only count as canceled operation if there was no fill
                        if filled_size == 0:
                            self.hourly_canceled_orders += 1
                        else:
                            # Partial fill, count as position operation
                            self.hourly_position_operations += 1

                        if self.order_filled_amount > 0:
                            self.logger.log_transaction(order_id, side, self.order_filled_amount, message.get('price'), status)

                    self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                    f"{message.get('size')} @ {message.get('price')}", "INFO")
                elif status == "PARTIALLY_FILLED":
                    self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                    f"{filled_size} @ {message.get('price')}", "INFO")
                else:
                    self.logger.log(f"[{order_type}] [{order_id}] {status} "
                                    f"{message.get('size')} @ {message.get('price')}", "INFO")

            except Exception as e:
                self.logger.log(f"Error handling order update: {e}", "ERROR")
                self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")

        # Setup order update handler
        self.exchange_client.setup_order_update_handler(order_update_handler)

    def _calculate_wait_time(self) -> Decimal:
        """Calculate wait time between orders."""
        cool_down_time = self.config.wait_time

        if len(self.active_close_orders) < self.last_close_orders:
            self.last_close_orders = len(self.active_close_orders)
            return 0

        self.last_close_orders = len(self.active_close_orders)
        if len(self.active_close_orders) >= self.config.max_orders:
            return 1

        if len(self.active_close_orders) / self.config.max_orders >= 2/3:
            cool_down_time = 2 * self.config.wait_time
        elif len(self.active_close_orders) / self.config.max_orders >= 1/3:
            cool_down_time = self.config.wait_time
        elif len(self.active_close_orders) / self.config.max_orders >= 1/6:
            cool_down_time = self.config.wait_time / 2
        else:
            cool_down_time = self.config.wait_time / 4

        # if the program detects active_close_orders during startup, it is necessary to consider cooldown_time
        if self.last_open_order_time == 0 and len(self.active_close_orders) > 0:
            self.last_open_order_time = time.time()

        if time.time() - self.last_open_order_time > cool_down_time:
            return 0
        else:
            return 1

    async def _place_and_monitor_open_order(self) -> bool:
        """Place an order and monitor its execution."""
        try:
            # Reset state before placing order
            self.order_filled_event.clear()
            self.current_order_status = 'OPEN'
            self.order_filled_amount = 0.0

            # Place the order
            order_result = await self.exchange_client.place_open_order(
                self.config.contract_id,
                self.config.quantity,
                self.config.direction
            )

            if not order_result.success:
                self.logger.log(f"Failed to place order: {order_result.error_message}", "ERROR")
                return False

            # Wait for fill or timeout
            if not self.order_filled_event.is_set():
                timeout_seconds = int(os.getenv('ORDER_TIMEOUT_SECONDS', 10))  # Default to 10 seconds
                try:
                    await asyncio.wait_for(self.order_filled_event.wait(), timeout=timeout_seconds)
                except asyncio.TimeoutError:
                    self.logger.log(f"[OPEN] Order {order_result.order_id} timed out after {timeout_seconds} seconds, trying to cancel order", "INFO")
                    pass

            # Handle order result
            return await self._handle_order_result(order_result)

        except Exception as e:
            self.logger.log(f"Error placing order: {e}", "ERROR")
            self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")
            return False

    async def _handle_order_result(self, order_result) -> bool:
        """Handle the result of an order placement."""
        order_id = order_result.order_id
        filled_price = order_result.price

        if self.order_filled_event.is_set():
            self.last_open_order_time = time.time()
            # Place close order
            close_side = self.config.close_order_side
            close_price = self._calculate_close_price(filled_price)

            close_order_result = await self.exchange_client.place_close_order(
                self.config.contract_id,
                self.config.quantity,
                close_price,
                close_side
            )

            if not close_order_result.success:
                self.logger.log(f"[CLOSE] Failed to place close order: {close_order_result.error_message}", "ERROR")

            return True

        else:
            self.order_canceled_event.clear()
            # Cancel the order if it's still open
            self.logger.log(f"[OPEN] [{order_id}] Order time out, trying to cancel order", "INFO")
            try:
                cancel_result = await self.exchange_client.cancel_order(order_id)
                if not cancel_result.success:
                    self.logger.log(f"[CLOSE] Failed to cancel order {order_id}: {cancel_result.error_message}", "ERROR")
                else:
                    self.current_order_status = "CANCELED"

            except Exception as e:
                self.logger.log(f"[CLOSE] Error canceling order {order_id}: {str(e)}", "ERROR")

            if self.config.exchange == "backpack":
                self.order_filled_amount = cancel_result.filled_size
            else:
                # Wait for cancel event or timeout
                if not self.order_canceled_event.is_set():
                    try:
                        await asyncio.wait_for(self.order_canceled_event.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        order_info = await self.exchange_client.get_order_info(order_id)
                        self.order_filled_amount = order_info.filled_size

            if self.order_filled_amount > 0:
                close_side = self.config.close_order_side
                close_price = self._calculate_close_price(filled_price)

                close_order_result = await self.exchange_client.place_close_order(
                    self.config.contract_id,
                    self.order_filled_amount,
                    close_price,
                    close_side
                )
                self.last_open_order_time = time.time()

                if not close_order_result.success:
                    self.logger.log(f"[CLOSE] Failed to place close order: {close_order_result.error_message}", "ERROR")

            return True

        return False

    async def _log_status_periodically(self):
        """Log status information periodically, including positions and hourly statistics."""
        if time.time() - self.last_log_time > 60 or self.last_log_time == 0:
            print("--------------------------------")
            try:
                # Get active orders
                active_orders = await self.exchange_client.get_active_orders(self.config.contract_id)

                # Filter close orders
                self.active_close_orders = []
                for order in active_orders:
                    if order.side == self.config.close_order_side:
                        self.active_close_orders.append({
                            'id': order.order_id,
                            'price': order.price,
                            'size': order.size
                        })

                # Get positions
                position_amt = await self.exchange_client.get_account_positions()

                # Calculate active closing amount
                active_close_amount = sum(
                    Decimal(order.get('size', 0))
                    for order in self.active_close_orders
                    if isinstance(order, dict)
                )

                self.logger.log(f"Current Position: {position_amt} | Active closing amount: {active_close_amount}")
                self.last_log_time = time.time()
                # Check for position mismatch
                if abs(position_amt - active_close_amount) > (2 * self.config.quantity):
                    error_message = f"Position mismatch detected - Current position: {position_amt}, Active closing amount: {active_close_amount}"
                    
                    self.logger.log(f"\n\nERROR: [{self.config.exchange.upper()}_{self.config.ticker.upper()}] {error_message}", "ERROR")
                    self.logger.log("###### ERROR ###### ERROR ###### ERROR ###### ERROR #####", "ERROR")
                    self.logger.log("Please manually rebalance your position and take-profit orders", "ERROR")
                    self.logger.log("请手动平衡当前仓位和正在关闭的仓位", "ERROR")
                    self.logger.log("###### ERROR ###### ERROR ###### ERROR ###### ERROR #####", "ERROR")

                    lark_token = os.getenv("LARK_TOKEN")
                    if lark_token:
                        try:
                            async with LarkBot(lark_token) as bot:
                                error_data = {
                                    "current_position": str(position_amt),
                                    "active_closing_amount": str(active_close_amount),
                                    "position_difference": str(abs(position_amt - active_close_amount)),
                                    "max_allowed_difference": str(2 * self.config.quantity)
                                }
                                
                                await bot.send_notification(
                                    notification_type="error",
                                    instance_id=self.config.instance_id,
                                    exchange=self.config.exchange,
                                    ticker=self.config.ticker,
                                    message="Position mismatch detected! Manual intervention required.",
                                    data=error_data
                                )
                        except Exception as e:
                            self.logger.log(f"Failed to send error notification: {e}", "ERROR")

                    if not self.shutdown_requested:
                        self.shutdown_requested = True

                    mismatch_detected = True
                else:
                    mismatch_detected = False

                # Send hourly statistics to Lark
                current_time = time.time()
                if current_time - self.last_hour_reset > 3600:  # 1 hour
                    # Calculate estimated fees
                    estimated_fees = self.hourly_position_operations * self.config.quantity * Decimal(0.001)  # Assuming 0.1% fee
                    
                    hourly_stats_message = (f"Hourly Statistics for instance {self.config.instance_id} - "
                                          f"Position operations: {self.hourly_position_operations}, "
                                          f"Closing operations: {self.hourly_closing_operations}, "
                                          f"Successful fills: {self.hourly_successful_fills}, "
                                          f"Canceled orders: {self.hourly_canceled_orders}")
                    
                    self.logger.log(hourly_stats_message, "INFO")
                    
                    lark_token = os.getenv("LARK_TOKEN")
                    if lark_token:
                        try:
                            async with LarkBot(lark_token) as bot:
                                stats_data = {
                                    "position_operations": self.hourly_position_operations,
                                    "closing_operations": self.hourly_closing_operations,
                                    "successful_fills": self.hourly_successful_fills,
                                    "canceled_orders": self.hourly_canceled_orders,
                                    "estimated_fees": str(estimated_fees),
                                    "current_position": str(position_amt),
                                    "active_closing_amount": str(active_close_amount),
                                    "hour_period": f"{datetime.fromtimestamp(self.last_hour_reset).strftime('%Y-%m-%d %H:%M')} - {datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M')}"
                                }
                                
                                await bot.send_notification(
                                    notification_type="hourly_stats",
                                    instance_id=self.config.instance_id,
                                    exchange=self.config.exchange,
                                    ticker=self.config.ticker,
                                    message=f"Hourly trading statistics for instance {self.config.instance_id}",
                                    data=stats_data
                                )
                        except Exception as e:
                            self.logger.log(f"Failed to send hourly stats notification: {e}", "ERROR")

                    # Reset statistics
                    self.hourly_position_operations = 0
                    self.hourly_closing_operations = 0
                    self.hourly_successful_fills = 0
                    self.hourly_canceled_orders = 0
                    self.last_hour_reset = current_time

                return mismatch_detected

            except Exception as e:
                self.logger.log(f"Error in periodic status check: {e}", "ERROR")
                self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")

            print("--------------------------------")

    async def _meet_grid_step_condition(self) -> bool:
        if self.active_close_orders:
            picker = min if self.config.direction == "buy" else max
            next_close_order = picker(self.active_close_orders, key=lambda o: o["price"])
            next_close_price = next_close_order["price"]

            best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
            if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
                raise ValueError("No bid/ask data available")

            if self.config.direction == "buy":
                new_order_close_price = best_ask * (1 + self.config.take_profit/100)
                if next_close_price / new_order_close_price > 1 + self.config.grid_step/100:
                    return True
                else:
                    return False
            elif self.config.direction == "sell":
                new_order_close_price = best_bid * (1 - self.config.take_profit/100)
                if new_order_close_price / next_close_price > 1 + self.config.grid_step/100:
                    return True
                else:
                    return False
            else:
                raise ValueError(f"Invalid direction: {self.config.direction}")
        else:
            return True

    def _calculate_close_price(self, filled_price: Decimal) -> Decimal:
        """Calculate the close price based on the filled price and take profit."""
        if self.config.close_order_side == 'sell':
            return filled_price * (1 + self.config.take_profit / 100)
        else:
            return filled_price * (1 - self.config.take_profit / 100)

    async def run(self):
        """Main trading loop."""
        try:
            self.config.contract_id, self.config.tick_size = await self.exchange_client.get_contract_attributes()

            # Capture the running event loop for thread-safe callbacks
            self.loop = asyncio.get_running_loop()
            # Connect to exchange
            await self.exchange_client.connect()

            # Send startup notification
            await self.send_startup_notification()

            # Main trading loop
            while not self.shutdown_requested:
                # Update active orders
                active_orders = await self.exchange_client.get_active_orders(self.config.contract_id)

                # Filter close orders
                self.active_close_orders = []
                for order in active_orders:
                    if order.side == self.config.close_order_side:
                        self.active_close_orders.append({
                            'id': order.order_id,
                            'price': order.price,
                            'size': order.size
                        })

                # Periodic logging
                mismatch_detected = await self._log_status_periodically()
                if not mismatch_detected:
                    wait_time = self._calculate_wait_time()

                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        meet_grid_step_condition = await self._meet_grid_step_condition()
                        if not meet_grid_step_condition:
                            await asyncio.sleep(1)
                            continue

                        await self._place_and_monitor_open_order()
                        self.last_close_orders += 1

        except KeyboardInterrupt:
            self.logger.log("Bot stopped by user")
            await self.graceful_shutdown("User interruption (Ctrl+C)")
        except Exception as e:
            self.logger.log(f"Critical error: {e}", "ERROR")
            await self.graceful_shutdown(f"Critical error: {e}")
            raise
        finally:
            # Ensure all connections are closed even if graceful shutdown fails
            try:
                await self.exchange_client.disconnect()
            except Exception as e:
                self.logger.log(f"Error disconnecting from exchange: {e}", "ERROR")
