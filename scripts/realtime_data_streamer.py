import asyncio
import websockets
import json
import pandas as pd
import sqlite3
from datetime import datetime
import logging
import requests

class RealTimeDataStreamer:
    def __init__(self):
        self.websocket_url = "wss://stream.binance.com:9443/ws/btcusdt@ticker"
        self.db_path = "data/realtime_bitcoin.db"
        self.setup_database()
        
    def setup_database(self):
        """Create SQLite database for real-time data storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realtime_prices (
                timestamp DATETIME,
                price REAL,
                volume REAL,
                price_change_24h REAL,
                price_change_percent_24h REAL,
                high_24h REAL,
                low_24h REAL
            )
        ''')
        conn.commit()
        conn.close()
        print("✅ Real-time database initialized")
        
    async def stream_live_prices(self):
        """Stream live Bitcoin prices and store in database"""
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                print("🔴 LIVE: Connected to Binance WebSocket")
                
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Extract relevant data
                    price_data = {
                        'timestamp': datetime.now(),
                        'price': float(data['c']),  # Current price
                        'volume': float(data['v']),  # 24h volume
                        'price_change_24h': float(data['P']),  # 24h price change %
                        'price_change_percent_24h': float(data['p']),  # 24h price change
                        'high_24h': float(data['h']),  # 24h high
                        'low_24h': float(data['l'])   # 24h low
                    }
                    
                    # Store in database
                    self.store_price_data(price_data)
                    
                    # Print live update
                    print(f"📊 BTC: ${price_data['price']:,.2f} | 24h: {price_data['price_change_24h']:+.2f}% | Vol: {price_data['volume']:,.0f}")
                    
                    # Wait 1 second before next update
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logging.error(f"WebSocket error: {e}")
            print(f"❌ WebSocket connection failed: {e}")
            # Fallback to REST API
            await self.fallback_to_rest_api()
            
    async def fallback_to_rest_api(self):
        """Fallback to REST API if WebSocket fails"""
        print("🔄 Falling back to REST API...")
        while True:
            try:
                response = requests.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT")
                data = response.json()
                
                price_data = {
                    'timestamp': datetime.now(),
                    'price': float(data['lastPrice']),
                    'volume': float(data['volume']),
                    'price_change_24h': float(data['priceChangePercent']),
                    'price_change_percent_24h': float(data['priceChange']),
                    'high_24h': float(data['highPrice']),
                    'low_24h': float(data['lowPrice'])
                }
                
                self.store_price_data(price_data)
                print(f"📊 BTC (REST): ${price_data['price']:,.2f} | 24h: {price_data['price_change_24h']:+.2f}%")
                
                await asyncio.sleep(10)  # REST API every 10 seconds
                
            except Exception as e:
                print(f"❌ REST API error: {e}")
                await asyncio.sleep(30)
            
    def store_price_data(self, price_data):
        """Store price data in SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO realtime_prices 
            (timestamp, price, volume, price_change_24h, price_change_percent_24h, high_24h, low_24h)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            price_data['timestamp'],
            price_data['price'],
            price_data['volume'],
            price_data['price_change_24h'],
            price_data['price_change_percent_24h'],
            price_data['high_24h'],
            price_data['low_24h']
        ))
        conn.commit()
        conn.close()
        
    def get_latest_price(self):
        """Get the most recent price from database"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM realtime_prices 
            ORDER BY timestamp DESC 
            LIMIT 1
        ''', conn)
        conn.close()
        return df.iloc[0] if not df.empty else None
        
    def get_price_history(self, hours=24):
        """Get price history for the last N hours"""
        conn = sqlite3.connect(self.db_path)
        cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
        df = pd.read_sql_query('''
            SELECT * FROM realtime_prices 
            WHERE timestamp > ?
            ORDER BY timestamp ASC
        ''', conn, params=(cutoff_time,))
        conn.close()
        return df
        
    def start_streaming(self):
        """Start the real-time streaming process"""
        print("🚀 Starting real-time Bitcoin data streaming...")
        asyncio.run(self.stream_live_prices())

if __name__ == "__main__":
    streamer = RealTimeDataStreamer()
    streamer.start_streaming()