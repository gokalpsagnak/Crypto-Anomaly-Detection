# Crypto Anomaly Detection System

An advanced anomaly detection system for cryptocurrency time-series data. This project implements a hybrid approach combining statistical methods (Z-score) and machine learning (Isolation Forest) to identify unusual trading patterns.

## Features

- **Data Collection**: Fetches historical cryptocurrency data from Binance API.
- **Preprocessing**: Cleans and normalizes time-series data.
- **Hybrid Anomaly Detection**:
  - **Statistical**: Uses Z-score to detect outliers in volume and price changes.
  - **Machine Learning**: Applies Isolation Forest algorithm to identify complex anomalies.
- **Visualization**: Generates plots to visualize anomalies in price and volume.
- **Alerting**: Sends Telegram notifications when anomalies are detected.

## Prerequisites

- Python 3.8+
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `python-binance`, `requests`, `python-dotenv`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SeniorProject
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with the following variables:
   ```env
   BINANCE_API_KEY=your_api_key
   BINANCE_API_SECRET=your_api_secret
   TELEGRAM_CHAT_ID=your_chat_id
   TELEGRAM_BOT_TOKEN=your_bot_token
   ```

## Usage

Run the main script to start the anomaly detection:

```bash
python main.py
```

The script will:
1. Fetch data for the last 100 days.
2. Detect anomalies using Z-score and Isolation Forest.
3. Print detected anomalies to the console.
4. Send a Telegram notification if anomalies are found.
5. Generate plots in the `plots/` directory.

## Configuration

You can configure the detection parameters in `config.py`:

- `SYMBOL`: Cryptocurrency symbol (default: 'BTCUSDT').
- `TIMEFRAME`: Time interval (default: '1d').
- `DAYS_TO_ANALYZE`: Number of days to fetch data for (default: 100).
- `ZSCORE_THRESHOLD`: Z-score threshold for statistical detection (default: 3.0).
- `ISOLATION_FOREST_CONTAMINATION`: Contamination parameter for Isolation Forest (default: 'auto').

## Project Structure

```
SeniorProject/
├── data/                    # Fetched cryptocurrency data
├── plots/                   # Generated visualizations
├── config.py                # Configuration settings
├── main.py                  # Main execution script
├── anomaly_detector.py      # Anomaly detection logic
├── data_fetcher.py          # Data fetching module
├── telegram_notifier.py     # Telegram notification module
├── requirements.txt         # Project dependencies
└── .env                     # Environment variables
```

## License

This project is licensed under the MIT License.
