import * as cdk from 'aws-cdk-lib';
import * as codebuild from 'aws-cdk-lib/aws-codebuild';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as cr from 'aws-cdk-lib/custom-resources';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';

export interface TritonImageBuilderProps extends cdk.StackProps {
    tritonEcrRepoName: string;
    trainingEcrRepoName: string;
    ngcApiKey: string;
    githubRepoUrl: string;
    githubBranch?: string;
    importExistingRepos?: boolean;
}

export class TritonImageBuilderStack extends cdk.Stack {
    constructor(scope: Construct, id: string, props: TritonImageBuilderProps) {
        super(scope, id, props);

        let tritonEcrRepo: ecr.IRepository;
        let trainingEcrRepo: ecr.IRepository;

        if (props.importExistingRepos) {
            tritonEcrRepo = ecr.Repository.fromRepositoryName(this, 'TritonEcrRepo', props.tritonEcrRepoName);
            trainingEcrRepo = ecr.Repository.fromRepositoryName(this, 'TrainingEcrRepo', props.trainingEcrRepoName);
        } else {
            tritonEcrRepo = new ecr.Repository(this, 'TritonEcrRepo', {
                repositoryName: props.tritonEcrRepoName,
                removalPolicy: cdk.RemovalPolicy.DESTROY,
                emptyOnDelete: true,
                imageScanOnPush: true,
                lifecycleRules: [{ maxImageCount: 5, description: 'Keep only the last 5 images' }],
            });
            trainingEcrRepo = new ecr.Repository(this, 'TrainingEcrRepo', {
                repositoryName: props.trainingEcrRepoName,
                removalPolicy: cdk.RemovalPolicy.DESTROY,
                emptyOnDelete: true,
                imageScanOnPush: true,
                lifecycleRules: [{ maxImageCount: 5, description: 'Keep only the last 5 images' }],
            });
        }

        const ngcApiKeySecret = new secretsmanager.Secret(this, 'NgcApiKey', {
            secretName: 'ngc-api-key',
            description: 'NGC API key for pulling NVIDIA container images',
            secretStringValue: cdk.SecretValue.unsafePlainText(props.ngcApiKey),
        });

        const buildProject = new codebuild.Project(this, 'TritonImageBuilder', {
            projectName: 'triton-fraud-detection-image-builder',
            description: 'Builds custom Triton inference server image with ML dependencies',
            source: codebuild.Source.gitHub({
                owner: props.githubRepoUrl.split('/').slice(-2)[0],
                repo: props.githubRepoUrl.split('/').slice(-1)[0].replace('.git', ''),
                branchOrRef: props.githubBranch || 'main',
            }),
            environment: {
                buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
                computeType: codebuild.ComputeType.LARGE,
                privileged: true,
                environmentVariables: {
                    NGC_API_KEY: { type: codebuild.BuildEnvironmentVariableType.SECRETS_MANAGER, value: ngcApiKeySecret.secretArn },
                    ECR_REGISTRY: { type: codebuild.BuildEnvironmentVariableType.PLAINTEXT, value: `${this.account}.dkr.ecr.${this.region}.amazonaws.com` },
                    TRITON_REPO: { type: codebuild.BuildEnvironmentVariableType.PLAINTEXT, value: props.tritonEcrRepoName },
                    TRAINING_REPO: { type: codebuild.BuildEnvironmentVariableType.PLAINTEXT, value: props.trainingEcrRepoName },
                },
            },
            timeout: cdk.Duration.hours(2),
            buildSpec: codebuild.BuildSpec.fromObject({
                version: '0.2',
                phases: {
                    pre_build: {
                        commands: [
                            'echo Logging in to Amazon ECR...',
                            'aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY',
                            'echo Logging in to NGC...',
                            "echo $NGC_API_KEY | docker login nvcr.io --username '\$oauthtoken' --password-stdin",
                        ],
                    },
                    build: {
                        commands: [
                            'echo Building Triton image...',
                            'docker build -t triton-fraud-detection:latest -f triton/Dockerfile triton/',
                            'docker tag triton-fraud-detection:latest $ECR_REGISTRY/$TRITON_REPO:latest',
                            'echo Pulling NVIDIA training image...',
                            'docker pull nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0',
                            'docker tag nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0 $ECR_REGISTRY/$TRAINING_REPO:latest',
                        ],
                    },
                    post_build: {
                        commands: [
                            'docker push $ECR_REGISTRY/$TRITON_REPO:latest',
                            'docker push $ECR_REGISTRY/$TRAINING_REPO:latest',
                            'echo Done!',
                        ],
                    },
                },
            }),
        });

        tritonEcrRepo.grantPullPush(buildProject);
        trainingEcrRepo.grantPullPush(buildProject);
        buildProject.addToRolePolicy(new iam.PolicyStatement({ actions: ['ecr:GetAuthorizationToken'], resources: ['*'] }));
        ngcApiKeySecret.grantRead(buildProject);

        new cr.AwsCustomResource(this, 'TriggerBuild', {
            onCreate: {
                service: 'CodeBuild',
                action: 'startBuild',
                parameters: { projectName: buildProject.projectName },
                physicalResourceId: cr.PhysicalResourceId.of(Date.now().toString()),
                outputPaths: ['build.id'],
            },
            onUpdate: {
                service: 'CodeBuild',
                action: 'startBuild',
                parameters: { projectName: buildProject.projectName },
                physicalResourceId: cr.PhysicalResourceId.of(Date.now().toString()),
                outputPaths: ['build.id'],
            },
            policy: cr.AwsCustomResourcePolicy.fromStatements([
                new iam.PolicyStatement({ actions: ['codebuild:StartBuild'], resources: [buildProject.projectArn] }),
            ]),
        });

        new cdk.CfnOutput(this, 'CodeBuildProjectName', { value: buildProject.projectName });
        new cdk.CfnOutput(this, 'TritonImageUri', { value: `${this.account}.dkr.ecr.${this.region}.amazonaws.com/${props.tritonEcrRepoName}:latest` });
        new cdk.CfnOutput(this, 'TrainingImageUri', { value: `${this.account}.dkr.ecr.${this.region}.amazonaws.com/${props.trainingEcrRepoName}:latest` });
    }
}
