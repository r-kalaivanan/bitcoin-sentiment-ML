#!/bin/bash

# Kubernetes deployment script for Bitcoin Sentiment ML Dashboard
# Usage: ./deploy_kubernetes.sh [namespace]

set -e

NAMESPACE=${1:-bitcoin-ml}
APP_NAME="bitcoin-sentiment-ml"
IMAGE_NAME="bitcoin-sentiment-ml"
VERSION=${VERSION:-latest}

echo "🚀 Deploying Bitcoin Sentiment ML to Kubernetes..."
echo "Namespace: $NAMESPACE"
echo "App Name: $APP_NAME"
echo "Image: $IMAGE_NAME:$VERSION"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Create ConfigMap for application configuration
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: $APP_NAME-config
  namespace: $NAMESPACE
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  ENABLE_CACHING: "true"
  MAX_CACHE_ENTRIES: "100"
  STREAMLIT_PORT: "8501"
EOF

# Create Secret for sensitive data
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: $APP_NAME-secrets
  namespace: $NAMESPACE
type: Opaque
data:
  # Add base64 encoded secrets here
  # BITCOIN_API_KEY: <base64-encoded-api-key>
EOF

# Create PersistentVolume for data storage
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: $APP_NAME-data-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: standard
EOF

# Create Deployment
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $APP_NAME
  namespace: $NAMESPACE
  labels:
    app: $APP_NAME
spec:
  replicas: 2
  selector:
    matchLabels:
      app: $APP_NAME
  template:
    metadata:
      labels:
        app: $APP_NAME
    spec:
      containers:
      - name: app
        image: $IMAGE_NAME:$VERSION
        ports:
        - containerPort: 8501
          name: http
        env:
        - name: STREAMLIT_PORT
          valueFrom:
            configMapKeyRef:
              name: $APP_NAME-config
              key: STREAMLIT_PORT
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: $APP_NAME-config
              key: ENVIRONMENT
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: $APP_NAME-config
              key: LOG_LEVEL
        - name: ENABLE_CACHING
          valueFrom:
            configMapKeyRef:
              name: $APP_NAME-config
              key: ENABLE_CACHING
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: logs-volume
          mountPath: /app/logs
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: $APP_NAME-data-pvc
      - name: logs-volume
        emptyDir: {}
      restartPolicy: Always
EOF

# Create Service
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: $APP_NAME-service
  namespace: $NAMESPACE
  labels:
    app: $APP_NAME
spec:
  selector:
    app: $APP_NAME
  ports:
  - port: 80
    targetPort: 8501
    protocol: TCP
    name: http
  type: ClusterIP
EOF

# Create Ingress
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: $APP_NAME-ingress
  namespace: $NAMESPACE
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "30"
    nginx.ingress.kubernetes.io/rate-limit-burst: "50"
spec:
  tls:
  - hosts:
    - bitcoin-ml.yourdomain.com
    secretName: $APP_NAME-tls
  rules:
  - host: bitcoin-ml.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: $APP_NAME-service
            port:
              number: 80
EOF

# Create HorizontalPodAutoscaler
cat <<EOF | kubectl apply -f -
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: $APP_NAME-hpa
  namespace: $NAMESPACE
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: $APP_NAME
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF

echo "✅ Deployment complete!"
echo ""
echo "📋 Deployment Summary:"
echo "Namespace: $NAMESPACE"
echo "Replicas: 2 (auto-scaling enabled)"
echo "Resources: 1-2 GB RAM, 0.5-1 CPU per pod"
echo ""
echo "🔍 Useful commands:"
echo "  Check status: kubectl get all -n $NAMESPACE"
echo "  View logs: kubectl logs -f deployment/$APP_NAME -n $NAMESPACE"
echo "  Scale manually: kubectl scale deployment $APP_NAME --replicas=3 -n $NAMESPACE"
echo "  Port forward: kubectl port-forward svc/$APP_NAME-service 8501:80 -n $NAMESPACE"
echo ""
echo "🌐 Access your application:"
echo "  Local (port-forward): http://localhost:8501"
echo "  External: https://bitcoin-ml.yourdomain.com"