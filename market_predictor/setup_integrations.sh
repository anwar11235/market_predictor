#!/bin/bash

# Create main integrations directory and subdirectories
mkdir -p src/integrations/{news,social,market}

# Create __init__.py files
touch src/integrations/__init__.py
touch src/integrations/news/__init__.py
touch src/integrations/social/__init__.py
touch src/integrations/market/__init__.py

# Create news integration files
touch src/integrations/news/newsapi_client.py
touch src/integrations/news/alpha_vantage_news.py
touch src/integrations/news/finnhub_client.py

# Create social integration files
touch src/integrations/social/twitter_client.py
touch src/integrations/social/reddit_client.py

# Create market integration files
touch src/integrations/market/finnhub_market.py

echo "Integration directory structure created successfully!"