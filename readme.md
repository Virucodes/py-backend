# Full-Stack Project Documentation

## Overview
This project consists of three main components:
- React Frontend Application
- Python Backend API
- Node.js Backend Service

## Table of Contents
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Applications](#running-the-applications)
- [API Documentation](#api-documentation)
- [Development Guidelines](#development-guidelines)
- [Deployment](#deployment)
- [Contributing](#contributing)

## System Requirements
### Frontend (React)
- Node.js (v18.0.0 or higher)
- npm (v8.0.0 or higher)
- Modern web browser

### Python Backend
- Python (v3.9 or higher)
- pip (latest version)
- Virtual environment (recommended)
- PostgreSQL (v13 or higher)

### Node Backend
- Node.js (v18.0.0 or higher)
- npm (v8.0.0 or higher)
- MongoDB (v5.0 or higher)

## Project Structure
```
project-root/
├── frontend/                # React application
│   ├── src/
│   ├── public/
│   └── package.json
├── python-backend/         # Python API service
│   ├── app/
│   ├── tests/
│   └── requirements.txt
└── node-backend/          # Node.js service
    ├── src/
    ├── tests/
    └── package.json
```

## Installation & Setup

### Frontend Setup
```bash
cd frontend
npm install
cp .env.example .env
# Configure your environment variables
```

### Python Backend Setup
```bash
cd python-backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Configure your environment variables
```

### Node Backend Setup
```bash
cd node-backend
npm install
cp .env.example .env
# Configure your environment variables
```

## Running the Applications

### Frontend Development Server
```bash
cd frontend
npm start
# Application will be available at http://localhost:3000
```

### Python Backend Server
```bash
cd python-backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python run.py
# API will be available at http://localhost:5000
```

### Node Backend Server
```bash
cd node-backend
npm run dev
# Service will be available at http://localhost:4000
```

## API Documentation

### Python Backend Endpoints
- `GET /api/v1/users` - Retrieve users list
- `POST /api/v1/users` - Create new user
- `GET /api/v1/users/<id>` - Retrieve specific user
- Full API documentation available at `/api/docs`

### Node Backend Endpoints
- `GET /api/v1/products` - Retrieve products list
- `POST /api/v1/products` - Create new product
- `GET /api/v1/products/<id>` - Retrieve specific product
- Swagger documentation available at `/api-docs`

## Development Guidelines

### Code Style
- Frontend: Follow ESLint configuration
- Python: Follow PEP 8 guidelines
- Node.js: Follow Airbnb style guide

### Git Workflow
1. Create feature branch from `develop`
2. Make changes and commit using conventional commits
3. Push changes and create PR
4. Ensure CI passes and get code review
5. Merge to `develop`

### Testing
- Frontend: Jest and React Testing Library
- Python: pytest
- Node.js: Jest

## Deployment

### Production Build Commands
```bash
# Frontend
cd frontend
npm run build

# Python Backend
cd python-backend
python -m pip install -r requirements.txt

# Node Backend
cd node-backend
npm run build
```

### Environment Variables
Each service requires specific environment variables. Check `.env.example` files in each directory for required variables.

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Support
For support, please contact the development team or create an issue in the repository.
