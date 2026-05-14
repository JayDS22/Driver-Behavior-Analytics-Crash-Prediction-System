#!/usr/bin/env bash
# Package the inference Lambda and deploy the CloudFormation stack.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
STACK_NAME="${STACK_NAME:-vehicle-safety-stack}"
ENV_NAME="${ENV_NAME:-prod}"
REGION="${AWS_REGION:-us-east-1}"
CODE_BUCKET="${CODE_BUCKET:-vehicle-safety-deploy-${ENV_NAME}}"
CODE_KEY="dist/inference-lambda-$(date +%Y%m%d%H%M%S).zip"
BUILD_DIR="${ROOT_DIR}/build/lambda"
ZIP_PATH="${ROOT_DIR}/build/inference-lambda.zip"

echo "[deploy] Packaging Lambda artifact..."
rm -rf "${BUILD_DIR}" "${ZIP_PATH}"
mkdir -p "${BUILD_DIR}"

cp -r "${ROOT_DIR}/src" "${BUILD_DIR}/"
cp -r "${ROOT_DIR}/config" "${BUILD_DIR}/"
cp -r "${ROOT_DIR}/deployment" "${BUILD_DIR}/"

pip install \
  --target "${BUILD_DIR}" \
  --no-cache-dir \
  -r "${ROOT_DIR}/requirements.txt"

(cd "${BUILD_DIR}" && zip -qr "${ZIP_PATH}" .)
echo "[deploy] Built ${ZIP_PATH}"

echo "[deploy] Ensuring deploy bucket exists: ${CODE_BUCKET}"
aws s3api head-bucket --bucket "${CODE_BUCKET}" 2>/dev/null \
  || aws s3 mb "s3://${CODE_BUCKET}" --region "${REGION}"

echo "[deploy] Uploading artifact to s3://${CODE_BUCKET}/${CODE_KEY}"
aws s3 cp "${ZIP_PATH}" "s3://${CODE_BUCKET}/${CODE_KEY}"

echo "[deploy] Deploying stack ${STACK_NAME} in ${REGION}"
aws cloudformation deploy \
  --region "${REGION}" \
  --stack-name "${STACK_NAME}" \
  --template-file "${ROOT_DIR}/deployment/aws/cloudformation.yaml" \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
      EnvironmentName="${ENV_NAME}" \
      LambdaCodeS3Bucket="${CODE_BUCKET}" \
      LambdaCodeS3Key="${CODE_KEY}"

aws cloudformation describe-stacks \
  --region "${REGION}" \
  --stack-name "${STACK_NAME}" \
  --query "Stacks[0].Outputs"
