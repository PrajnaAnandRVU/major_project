# FinSense AI

FinSense AI is a modern financial analysis and sentiment analysis platform built with React, TypeScript, and Python. The platform provides real-time financial data analysis, market sentiment insights, and interactive data visualization.

## Features

- Real-time financial data analysis
- Market sentiment analysis
- Interactive data visualization using ECharts
- Modern, responsive UI built with React and Tailwind CSS
- Type-safe development with TypeScript
- Fast development and build times with Vite

## Tech Stack

### Frontend
- React 18
- TypeScript
- Tailwind CSS
- ECharts for data visualization
- Axios for API requests
- Vite for build tooling

### Backend
- Python
- Flask/FastAPI (based on the backend directory structure)

## Prerequisites

- Node.js (v16 or higher)
- Python 3.8 or higher
- npm or yarn package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/adrocx/FinSense_AI_.git
cd FinSense_AI_
```

2. Install frontend dependencies:
```bash
npm install
```

3. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

## Development

1. Start the frontend development server:
```bash
npm run dev
```

2. Start the backend server (from the backend directory):
```bash
python app.py  # or the appropriate command based on your backend setup
```

The frontend will be available at `http://localhost:5173` and the backend API at `http://localhost:5000` (or the configured port).

## Building for Production

To create a production build:

```bash
npm run build
```

The build artifacts will be stored in the `dist/` directory.

## Project Structure

```
finsense-react/
├── src/              # Frontend source code
├── backend/          # Python backend code
├── public/           # Static assets
├── node_modules/     # Frontend dependencies
└── package.json      # Frontend dependencies and scripts
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Your Name - Aditya Raju

Project Link: [https://github.com/adrocx/FinSense_AI_](https://github.com/adrocx/FinSense_AI_)
