#!/usr/bin/env node
import * as dotenv from "dotenv";
dotenv.config(); // Load .env file

import * as cdk from "aws-cdk-lib";
import { AwsSolutionsChecks } from "cdk-nag";
import { NvidiaFraudDetectionBlueprint } from "../lib/nvidia-fraud-detection-blueprint";
import { TarExtractorStack } from "../lib/tar-extractor-stack";
import { SageMakerExecutionRoleStack } from "../lib/sagemaker-training-role";
import { SageMakerNotebookRoleStack } from "../lib/sagemaker-notebook-role";
import { TritonImageBuilderStack } from "../lib/triton-image-builder";

const app = new cdk.App();

// Add CDK Nag checks
//cdk.Aspects.of(app).add(new AwsSolutionsChecks({ verbose: true }));

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION,
};

const modelBucketName = "ml-on-containers-" + process.env.CDK_DEFAULT_ACCOUNT;

const tarExtractorStack = new TarExtractorStack(
  app,
  "NvidiaFraudDetectionBlueprintModelExtractor",
  {
    env: env,
    modelBucketName: modelBucketName,
  },
);

const sagemakerExecutionRole = new SageMakerExecutionRoleStack(
  app,
  "NvidiaFraudDetectionTrainingRole",
  {
    env: env,
    modelBucketArn: "arn:aws:s3:::" + modelBucketName,
  },
);

const sagemakerNotebookRole = new SageMakerNotebookRoleStack(
  app,
  "NvidiaFraudDetectionNotebookRole",
  {
    env: env,
    modelBucketArn: "arn:aws:s3:::" + modelBucketName,
    modelRegistryBucketArn: "arn:aws:s3:::" + modelBucketName + "-model-registry",
  },
);

const mainStack = new NvidiaFraudDetectionBlueprint(
  app,
  "NvidiaFraudDetectionBlueprint",
  {
    env: env,
    modelBucketName: modelBucketName + "-model-registry",
  },
);

// Triton inference server and training image builder
// Requires NGC_API_KEY environment variable (from .env file)
const ngcApiKey = process.env.NGC_API_KEY;
if (!ngcApiKey) {
  throw new Error("NGC_API_KEY environment variable is required. Set it in .env file.");
}

const tritonImageBuilder = new TritonImageBuilderStack(
  app,
  "NvidiaFraudDetectionImageBuilder",
  {
    env: env,
    tritonEcrRepoName: "triton-fraud-detection",
    trainingEcrRepoName: "nvidia-training-repo",
    ngcApiKey: ngcApiKey,
    githubRepoUrl: "https://github.com/atroyanovsky/TW-sample-financial-fraud-detection-with-nvidia",
    githubBranch: "main",
    importExistingRepos: true, // Set to false for fresh deployments
  },
);

mainStack.addDependency(sagemakerExecutionRole);
mainStack.addDependency(sagemakerNotebookRole);
mainStack.addDependency(tarExtractorStack);
mainStack.addDependency(tritonImageBuilder);
