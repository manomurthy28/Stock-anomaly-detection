# Stock Anomaly Detection Dashboard

This project is a dashboard for detecting and visualizing anomalies in stock market data using two different machine learning models: Isolation Forest and LSTM Autoencoder.

uvicorn api:app --host 0.0.0.0 --port 8000 --reload

## Features

- Enter a stock symbol to analyze historical price data
- View detected anomalies from both models on a price chart
- Compare model performance metrics
- Visualize feature importance for each model
- See model agreement on detected anomalies
- Store and retrieve anomalies in a PostgreSQL database
- View system logs and user activity
- Real-time notifications for newly detected anomalies
- Watch specific stocks for anomaly alerts

## Project Structure

\`\`\`
stockanomalydashboard/
├── app/                  # Next.js app directory
│   ├── api/              # API routes for database operations
│   ├── anomalies/        # Anomaly history page
│   ├── logs/             # System logs page
│   └── page.tsx          # Main dashboard page
├── components/           # React components
├── contexts/             # React contexts (notification)
├── lib/                  # Utility functions and services
│   ├── api.ts            # API client functions
│   ├── anomaly-service.ts # Anomaly storage and retrieval
│   ├── logger.ts         # Logging service
│   └── prisma.ts         # Prisma client
├── prisma/               # Prisma ORM
│   └── schema.prisma     # Database schema
├── scripts/              # Python scripts and database initialization
│   ├── api.py            # FastAPI server
│   ├── isolation_forest.py  # Isolation Forest model
│   ├── lstm_autoencoder.py  # LSTM Autoencoder model
│   ├── stock_analyzer.py    # Main analysis logic
│   ├── requirements.txt     # Python dependencies
│   └── init-db.ts        # Database initialization script
├── backend/              # Python backend
│   ├── main.py           # FastAPI server
│   └── requirements.txt  # Python dependencies
└── public/               # Static assets
\`\`\`

## Setup Instructions

### Prerequisites

- Node.js 16+ and npm
- PostgreSQL 12+
- Python 3.8+
- Git

### Step 1: Clone the Repository

\`\`\`bash
git clone https://github.com/yourusername/stockanomalydashboard.git
cd stockanomalydashboard
\`\`\`

### Step 2: Install JavaScript Dependencies

\`\`\`bash
npm install
\`\`\`

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root:

\`\`\`bash
# Database connection
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/stock_anomaly_dashboard?schema=public"

# API Base URL
NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"
\`\`\`

Adjust the PostgreSQL connection string if your credentials or port are different.

### Step 4: Create the Database

\`\`\`bash
# For PostgreSQL
createdb stock_anomaly_dashboard

# Or using psql
psql -U postgres -c "CREATE DATABASE stock_anomaly_dashboard;"
\`\`\`

### Step 5: Set Up the Database with Prisma

\`\`\`bash
# Generate Prisma client
npx prisma generate

# Push schema to database
npx prisma db push

# Seed the database
npx ts-node --compiler-options '{"module":"CommonJS"}' scripts/init-db.ts
\`\`\`

### Step 6: Set Up Python Backend

\`\`\`bash
# Navigate to the backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt
\`\`\`

## Running the Application

### Step 1: Start the Next.js Development Server

In the project root directory:

\`\`\`bash
npm run dev
\`\`\`

This will start the Next.js server at http://localhost:3000.

### Step 2: Start the Python Backend

In a separate terminal, navigate to the backend directory:

\`\`\`bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
\`\`\`

This will start the FastAPI server at http://localhost:8000.

### Step 3: Access the Dashboard

Open your browser and navigate to http://localhost:3000 to access the dashboard.

### Step 4: Explore the Database (Optional)

You can use Prisma Studio to explore and manage your database:

\`\`\`bash
npx prisma studio
\`\`\`

This will open a web interface at http://localhost:5555 where you can view and edit your database.

## Complete Setup Script

Here's a shell script that performs all the setup steps:

\`\`\`bash
#!/bin/bash

# Install dependencies
npm install

# Generate Prisma client
npx prisma generate

# Push schema to database
npx prisma db push

# Seed the database
npx ts-node --compiler-options '{"module":"CommonJS"}' scripts/init-db.ts

# Start the development server
echo "Setup complete! Run 'npm run dev' to start the Next.js server"
echo "In a separate terminal, run 'cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload' to start the Python API"
\`\`\`

Save this as `setup.sh`, make it executable with `chmod +x setup.sh`, and run it with `./setup.sh`.

## Using the Dashboard

1. Enter a stock symbol (e.g., AAPL, MSFT, GOOGL) in the search box and click "Analyze"
2. View the detected anomalies on the price chart
3. Explore the different tabs to see model comparisons and performance metrics
4. Click the "Watch" button to receive notifications for new anomalies
5. Navigate to the "Anomalies" page to view historical anomalies
6. Check the "Logs" page to see system events and user activity

## Troubleshooting

### Database Connection Error

If you encounter database connection errors:

1. Make sure PostgreSQL is running
2. Verify the connection string in `.env` is correct
3. Check that the database exists
4. Ensure Prisma has been properly set up with `npx prisma generate`

### Python API Error

If the Python API is not working:

1. Make sure all Python dependencies are installed
2. Check that the API is running on port 8000
3. Verify the `NEXT_PUBLIC_API_BASE_URL` in `.env` is correct
4. Look for error messages in the terminal running the Python API

### Port Conflicts

If port 3000 (Next.js) or 8000 (Python API) are already in use, you can specify different ports:

\`\`\`bash
# For Next.js
npm run dev -- -p 3001

# For Python API
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
\`\`\`

Remember to update the `NEXT_PUBLIC_API_BASE_URL` in `.env` if you change the Python API port.

## Technologies Used

- **Frontend**: Next.js, React, Tailwind CSS, Chart.js
- **Backend**: FastAPI, TensorFlow, scikit-learn, pandas
- **Database**: PostgreSQL, Prisma ORM
- **Data**: yfinance for stock data retrieval

## License

This project is licensed under the MIT License - see the LICENSE file for details.
