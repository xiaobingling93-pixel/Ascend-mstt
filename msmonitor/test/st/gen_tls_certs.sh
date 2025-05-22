#!/bin/bash

set -e
set -o pipefail

# Configuration
CERTS_DIR="certs"
DAYS_VALID=3650
SERVER_CN="localhost"
SERVER_SAN="DNS:localhost,IP:127.0.0.1"
CLIENT_CN="client"

# Check if openssl is installed
if ! command -v openssl &> /dev/null; then
    echo "Error: openssl is not installed. Please install it first."
    exit 1
fi

# Create certs directory
mkdir -p "$CERTS_DIR"

# Generate CA root certificate and private key
echo "Generating CA root certificate and private key..."
openssl genrsa -out "$CERTS_DIR/ca.key" 2048
openssl req -x509 -new -key "$CERTS_DIR/ca.key" -days "$DAYS_VALID" -out "$CERTS_DIR/ca.crt" -subj "/CN=Test-CA"

# Generate client private key and CSR
echo "Generating client private key and CSR..."
openssl genrsa -out "$CERTS_DIR/client.key" 2048
openssl req -new -key "$CERTS_DIR/client.key" -out "$CERTS_DIR/client.csr" -subj "/CN=$CLIENT_CN"

# Create v3.ext for client certificate
echo "Creating v3.ext for client certificate..."
cat > "$CERTS_DIR/v3.ext" <<EOF
[ v3_req ]
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = clientAuth
subjectAltName = @alt_names

[ alt_names ]
DNS.1 = $CLIENT_CN
EOF

# Sign client certificate
echo "Signing client certificate..."
openssl x509 -req -in "$CERTS_DIR/client.csr" -CA "$CERTS_DIR/ca.crt" -CAkey "$CERTS_DIR/ca.key" -CAcreateserial \
  -out "$CERTS_DIR/client.crt" -days "$DAYS_VALID" -sha256 -extensions v3_req -extfile "$CERTS_DIR/v3.ext"

# Generate server private key
echo "Generating server private key..."
openssl genrsa -out "$CERTS_DIR/server.key" 2048

# Create server.ext for server certificate
echo "Creating server.ext for server certificate..."
cat > "$CERTS_DIR/server.ext" <<EOF
[ req ]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[ req_distinguished_name ]
CN = $SERVER_CN

[ v3_req ]
subjectAltName = $SERVER_SAN
EOF

# Generate server CSR
echo "Generating server CSR..."
openssl req -new -key "$CERTS_DIR/server.key" -out "$CERTS_DIR/server.csr" -subj "/CN=$SERVER_CN" -config "$CERTS_DIR/server.ext"

# Sign server certificate
echo "Signing server certificate..."
openssl x509 -req -in "$CERTS_DIR/server.csr" -CA "$CERTS_DIR/ca.crt" -CAkey "$CERTS_DIR/ca.key" -CAcreateserial \
  -out "$CERTS_DIR/server.crt" -days "$DAYS_VALID" -sha256 -extfile "$CERTS_DIR/server.ext" -extensions v3_req

# Verify certificates
echo "Verifying certificates..."
openssl verify -CAfile "$CERTS_DIR/ca.crt" "$CERTS_DIR/client.crt"
openssl verify -CAfile "$CERTS_DIR/ca.crt" "$CERTS_DIR/server.crt"

# Display certificate info
echo "Client certificate:"
openssl x509 -in "$CERTS_DIR/client.crt" -text -noout | grep "Version:"
echo "Server certificate:"
openssl x509 -in "$CERTS_DIR/server.crt" -text -noout | grep "Version:"

# Cleanup temporary files
rm -f "$CERTS_DIR"/*.csr "$CERTS_DIR"/*.srl "$CERTS_DIR"/*.ext

echo "All certificates have been successfully generated in the $CERTS_DIR directory."