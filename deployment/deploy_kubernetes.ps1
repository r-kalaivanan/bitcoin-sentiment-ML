# PowerShell deployment script for Kubernetes
# Usage: .\deploy_kubernetes.ps1 [namespace]

param(
    [string]$Namespace = "bitcoin-ml",
    [string]$Version = "latest"
)

$APP_NAME = "bitcoin-sentiment-ml"
$IMAGE_NAME = "bitcoin-sentiment-ml"

Write-Host "🚀 Deploying Bitcoin Sentiment ML to Kubernetes..." -ForegroundColor Green
Write-Host "Namespace: $Namespace" -ForegroundColor Cyan
Write-Host "App Name: $APP_NAME" -ForegroundColor Cyan
Write-Host "Image: ${IMAGE_NAME}:$Version" -ForegroundColor Cyan

# Check if kubectl is available
try {
    kubectl version --client=true 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "kubectl not found"
    }
}
catch {
    Write-Error "❌ kubectl is not installed or not in PATH. Please install kubectl first."
    exit 1
}

# Create namespace if it doesn't exist
Write-Host "📁 Creating namespace..." -ForegroundColor Yellow
kubectl create namespace $Namespace --dry-run=client -o yaml | kubectl apply -f -

# Create ConfigMap
Write-Host "⚙️ Creating ConfigMap..." -ForegroundColor Yellow
@"
apiVersion: v1
kind: ConfigMap
metadata:
  name: $APP_NAME-config
  namespace: $Namespace
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  ENABLE_CACHING: "true"
  MAX_CACHE_ENTRIES: "100"
  STREAMLIT_PORT: "8501"
"@ | kubectl apply -f -

# Create Secret
Write-Host "🔐 Creating Secret..." -ForegroundColor Yellow
@"
apiVersion: v1
kind: Secret
metadata:
  name: $APP_NAME-secrets
  namespace: $Namespace
type: Opaque
data:
  # Add base64 encoded secrets here
  # BITCOIN_API_KEY: <base64-encoded-api-key>
"@ | kubectl apply -f -

# Create PersistentVolumeClaim
Write-Host "💾 Creating PersistentVolumeClaim..." -ForegroundColor Yellow
@"
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: $APP_NAME-data-pvc
  namespace: $Namespace
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: standard
"@ | kubectl apply -f -

# Create Deployment
Write-Host "🚢 Creating Deployment..." -ForegroundColor Yellow
@"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: $APP_NAME
  namespace: $Namespace
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
        image: ${IMAGE_NAME}:$Version
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
"@ | kubectl apply -f -

# Create Service
Write-Host "🌐 Creating Service..." -ForegroundColor Yellow
@"
apiVersion: v1
kind: Service
metadata:
  name: $APP_NAME-service
  namespace: $Namespace
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
"@ | kubectl apply -f -

# Create Ingress
Write-Host "🚪 Creating Ingress..." -ForegroundColor Yellow
@"
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: $APP_NAME-ingress
  namespace: $Namespace
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
"@ | kubectl apply -f -

# Create HorizontalPodAutoscaler
Write-Host "📈 Creating HorizontalPodAutoscaler..." -ForegroundColor Yellow
@"
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: $APP_NAME-hpa
  namespace: $Namespace
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
"@ | kubectl apply -f -

Write-Host "✅ Deployment complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Deployment Summary:" -ForegroundColor Cyan
Write-Host "Namespace: $Namespace" -ForegroundColor White
Write-Host "Replicas: 2 (auto-scaling enabled)" -ForegroundColor White
Write-Host "Resources: 1-2 GB RAM, 0.5-1 CPU per pod" -ForegroundColor White
Write-Host ""
Write-Host "🔍 Useful commands:" -ForegroundColor Cyan
Write-Host "  Check status: kubectl get all -n $Namespace" -ForegroundColor Yellow
Write-Host "  View logs: kubectl logs -f deployment/$APP_NAME -n $Namespace" -ForegroundColor Yellow
Write-Host "  Scale manually: kubectl scale deployment $APP_NAME --replicas=3 -n $Namespace" -ForegroundColor Yellow
Write-Host "  Port forward: kubectl port-forward svc/$APP_NAME-service 8501:80 -n $Namespace" -ForegroundColor Yellow
Write-Host ""
Write-Host "🌐 Access your application:" -ForegroundColor Cyan
Write-Host "  Local (port-forward): http://localhost:8501" -ForegroundColor Yellow
Write-Host "  External: https://bitcoin-ml.yourdomain.com" -ForegroundColor Yellow