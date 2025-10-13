# Bitcoin Sentiment ML Dashboard - Deployment Guide

## Overview

This comprehensive guide provides multiple deployment options for the Bitcoin Sentiment ML Dashboard, ranging from local development to production cloud deployments.

## 🚀 Quick Start Options

### Option 1: Docker (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t bitcoin-sentiment-ml .
docker run -p 8501:8501 bitcoin-sentiment-ml
```

### Option 2: Streamlit Cloud (Easiest)

1. Fork this repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy directly from your repository

### Option 3: Local Development

```bash
pip install -r requirements.txt
streamlit run launch_dashboard.py
```

## 📋 Prerequisites

### System Requirements

- Python 3.8+ (3.11 recommended)
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection for Bitcoin price updates

### Required Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=1.7.0
plotly>=5.15.0
joblib>=1.3.0
requests>=2.31.0
```

## 🐳 Docker Deployment

### Development Mode

```bash
# Clone repository
git clone https://github.com/your-username/bitcoin-sentiment-ml.git
cd bitcoin-sentiment-ml

# Build and run
docker-compose up --build
```

### Production Mode

```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up --build -d

# Check health
curl http://localhost:8501/health
```

### Docker Configuration Options

- **Port**: Default 8501 (configurable via STREAMLIT_PORT)
- **Environment**: Set ENVIRONMENT=production for optimizations
- **Logging**: Logs saved to `./logs/` directory
- **Data Persistence**: Data stored in `./data/` volume

## ☁️ Cloud Platform Deployments

### Streamlit Cloud

**Best for**: Quick prototypes, demos, free hosting

1. **Setup**:

   ```bash
   # Ensure you have these files:
   # - requirements.txt
   # - launch_dashboard.py
   # - .streamlit/config.toml
   ```

2. **Deploy**:

   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select repository and branch
   - Set main file: `launch_dashboard.py`
   - Deploy

3. **Configuration**:
   ```toml
   # .streamlit/config.toml
   [server]
   headless = true
   port = $PORT
   enableCORS = false
   enableXsrfProtection = false
   ```

### Heroku

**Best for**: Scalable hosting, custom domains, production apps

1. **Automatic Deployment**:

   ```bash
   # Run deployment script
   ./deployment/deploy_heroku.sh
   ```

2. **Manual Deployment**:

   ```bash
   # Install Heroku CLI
   heroku login
   heroku create your-app-name

   # Deploy
   git push heroku main
   ```

3. **Configuration Files**:
   - `Procfile`: `web: streamlit run launch_dashboard.py --server.port=$PORT`
   - `app.json`: App metadata and buildpack configuration
   - `runtime.txt`: Python version specification

### Railway

**Best for**: Modern deployment, automatic HTTPS, great developer experience

1. **Automatic Deployment**:

   ```bash
   # Run deployment script
   ./deployment/deploy_railway.sh
   ```

2. **Manual Deployment**:

   - Visit [railway.app](https://railway.app)
   - Connect GitHub repository
   - Configure build settings
   - Deploy

3. **Configuration**:
   ```json
   // railway.json
   {
     "build": {
       "builder": "DOCKERFILE"
     },
     "deploy": {
       "startCommand": "streamlit run launch_dashboard.py --server.port=$PORT"
     }
   }
   ```

### AWS Deployment

**Best for**: Enterprise applications, custom infrastructure, high traffic

1. **EC2 Deployment**:

   ```bash
   # Launch EC2 instance (t3.medium recommended)
   # Install Docker
   sudo yum update -y
   sudo yum install -y docker
   sudo service docker start

   # Deploy application
   git clone https://github.com/your-username/bitcoin-sentiment-ml.git
   cd bitcoin-sentiment-ml
   docker-compose up -d
   ```

2. **ECS Deployment**:

   ```bash
   # Build and push to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ECR_URI
   docker build -t bitcoin-sentiment-ml .
   docker tag bitcoin-sentiment-ml:latest YOUR_ECR_URI/bitcoin-sentiment-ml:latest
   docker push YOUR_ECR_URI/bitcoin-sentiment-ml:latest
   ```

3. **Elastic Beanstalk**:
   - Create `Dockerrun.aws.json`
   - Deploy via EB CLI or AWS Console

## 🔧 Configuration Options

### Environment Variables

```bash
# Core settings
ENVIRONMENT=production|development
STREAMLIT_PORT=8501
LOG_LEVEL=INFO|DEBUG|WARNING|ERROR

# Data sources
BITCOIN_API_KEY=your_api_key
DATA_UPDATE_INTERVAL=300

# Model settings
MODEL_CACHE_TTL=600
MAX_PREDICTION_HISTORY=1000

# Performance
ENABLE_CACHING=true
MAX_CACHE_ENTRIES=100
MEMORY_LIMIT_MB=512
```

### Streamlit Configuration

```toml
# .streamlit/config.toml
[server]
headless = true
port = 8501
maxUploadSize = 50
enableCORS = false

[theme]
primaryColor = "#F7931A"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[browser]
gatherUsageStats = false
```

### Production Optimizations

- **Caching**: Data and model caching with TTL
- **Memory Management**: Optimized DataFrame storage
- **Error Handling**: Comprehensive error logging
- **Health Checks**: Built-in health monitoring
- **Performance Monitoring**: Function execution tracking

## 🔍 Monitoring and Maintenance

### Health Checks

```bash
# Docker health check
curl http://localhost:8501/health

# Application status
curl http://localhost:8501/_stcore/health
```

### Logging

```bash
# View application logs
docker logs bitcoin-sentiment-ml-app-1

# Follow logs in real-time
docker logs -f bitcoin-sentiment-ml-app-1

# Log files location
./logs/app.log
```

### Performance Monitoring

- CPU and memory usage tracking
- Response time monitoring
- Error rate tracking
- Model prediction accuracy

## 🛠️ Troubleshooting

### Common Issues

#### Port Already in Use

```bash
# Find process using port 8501
lsof -i :8501
# Kill process
kill -9 <PID>
```

#### Memory Issues

```bash
# Increase Docker memory limit
docker run -m 2g -p 8501:8501 bitcoin-sentiment-ml
```

#### Model Loading Errors

```bash
# Check model files exist
ls -la models/
# Verify model integrity
python -c "import joblib; print(joblib.load('models/lightgbm_sentiment_enhanced.pkl'))"
```

#### Data Loading Issues

```bash
# Verify data files
ls -la data/
# Check data format
head -5 data/btc_data.csv
```

### Debug Mode

```bash
# Run with debug logging
export LOG_LEVEL=DEBUG
streamlit run launch_dashboard.py

# Or with Docker
docker run -e LOG_LEVEL=DEBUG -p 8501:8501 bitcoin-sentiment-ml
```

### Performance Issues

1. **Enable caching**: Set `ENABLE_CACHING=true`
2. **Reduce data size**: Limit historical data range
3. **Optimize models**: Use lighter model versions
4. **Increase resources**: Scale up container/instance

## 🔒 Security Considerations

### Production Security

- Remove debug flags in production
- Use environment variables for secrets
- Implement rate limiting
- Enable HTTPS (handled by cloud platforms)
- Regular security updates

### Environment Variables Security

```bash
# Use secret management
export BITCOIN_API_KEY=$(cat /run/secrets/api_key)
export DATABASE_URL=$(cat /run/secrets/db_url)
```

## 📊 Scaling Options

### Horizontal Scaling

- Multiple container instances
- Load balancer configuration
- Database connection pooling

### Vertical Scaling

- Increase CPU/memory resources
- Optimize model loading
- Implement model caching

### Auto-scaling

- Container orchestration (Kubernetes)
- Cloud auto-scaling groups
- Performance-based scaling triggers

## 🚀 Deployment Checklist

### Pre-deployment

- [ ] Test application locally
- [ ] Verify all dependencies in requirements.txt
- [ ] Check model files are included
- [ ] Configure environment variables
- [ ] Set up monitoring/logging

### Deployment

- [ ] Choose deployment platform
- [ ] Configure platform-specific settings
- [ ] Deploy application
- [ ] Verify health endpoints
- [ ] Test core functionality

### Post-deployment

- [ ] Monitor application performance
- [ ] Set up alerts for errors
- [ ] Plan backup/recovery strategy
- [ ] Document maintenance procedures
- [ ] Schedule regular updates

## 📞 Support

### Getting Help

- **GitHub Issues**: [Create an issue](https://github.com/r-kalaivanan/bitcoin-sentiment-ML/issues)
- **Documentation**: Check README.md for project details
- **Community**: Streamlit community forums

### Reporting Bugs

1. Check existing issues
2. Provide reproduction steps
3. Include system information
4. Attach relevant logs

## 🔄 Updates and Maintenance

### Regular Maintenance

- Update dependencies monthly
- Monitor model performance
- Review and rotate logs
- Security patch updates

### Model Updates

- Retrain models with new data
- A/B test new model versions
- Gradual rollout of updates
- Performance comparison tracking

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Maintainer**: Bitcoin Sentiment ML Team
