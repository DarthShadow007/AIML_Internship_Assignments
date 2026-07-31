# Assignment 8: End to End Render Deployment Project

## Overview
This project sets up an HTTP microservice architecture designed for deploying machine learning model endpoints on Render cloud infrastructure.

## Deployment Specifications
* **Protocol**: RESTful HTTP API
* **Environment**: Render Cloud Engine
* **Endpoints**:
  * `GET /health` - Microservice health check
  * `POST /predict` - Model inference execution

## How to Run
```bash
python 08_render_deployment.py