"""
ModelServe — Pulumi Infrastructure
Region: ap-southeast-1
Topology: Option A — All services on single EC2 instance (t3.small)
Resources: VPC, Subnet, IGW, Route Table, Security Group, Key Pair,
           IAM Role, S3 Bucket, ECR Repository, EC2 Instance, Elastic IP
"""

import os
import textwrap
import pulumi
import pulumi_aws as aws

# ── Tags applied to every resource ────────────────────────────────────────────
TAGS = {"Project": "modelserve"}

# ── GitHub repo URL (EC2 user-data clones this) ───────────────────────────────
GITHUB_REPO = "https://github.com/rafilovestosuffer/MLOps-S2-E1-modelserve-capstone-starter.git"

# ── Ubuntu 22.04 LTS AMI for ap-southeast-1 ──────────────────────────────────
# Using aws.ec2.get_ami to always fetch the latest Ubuntu 22.04 LTS
ubuntu_ami = aws.ec2.get_ami(
    most_recent=True,
    owners=["099720109477"],  # Canonical
    filters=[
        aws.ec2.GetAmiFilterArgs(name="name", values=["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]),
        aws.ec2.GetAmiFilterArgs(name="virtualization-type", values=["hvm"]),
        aws.ec2.GetAmiFilterArgs(name="state", values=["available"]),
    ],
)

# ══════════════════════════════════════════════════════════════════════════════
# NETWORKING
# ══════════════════════════════════════════════════════════════════════════════

vpc = aws.ec2.Vpc(
    "modelserve-vpc",
    cidr_block="10.0.0.0/16",
    enable_dns_hostnames=True,
    enable_dns_support=True,
    tags={**TAGS, "Name": "modelserve-vpc"},
)

subnet = aws.ec2.Subnet(
    "modelserve-subnet",
    vpc_id=vpc.id,
    cidr_block="10.0.1.0/24",
    availability_zone="ap-southeast-1a",
    map_public_ip_on_launch=True,
    tags={**TAGS, "Name": "modelserve-public-subnet"},
)

igw = aws.ec2.InternetGateway(
    "modelserve-igw",
    vpc_id=vpc.id,
    tags={**TAGS, "Name": "modelserve-igw"},
)

route_table = aws.ec2.RouteTable(
    "modelserve-rt",
    vpc_id=vpc.id,
    routes=[
        aws.ec2.RouteTableRouteArgs(
            cidr_block="0.0.0.0/0",
            gateway_id=igw.id,
        )
    ],
    tags={**TAGS, "Name": "modelserve-rt"},
)

aws.ec2.RouteTableAssociation(
    "modelserve-rta",
    subnet_id=subnet.id,
    route_table_id=route_table.id,
)

# ══════════════════════════════════════════════════════════════════════════════
# SECURITY GROUP
# ══════════════════════════════════════════════════════════════════════════════

sg = aws.ec2.SecurityGroup(
    "modelserve-sg",
    vpc_id=vpc.id,
    description="ModelServe security group — API, MLflow, Prometheus, Grafana, SSH",
    ingress=[
        # SSH
        aws.ec2.SecurityGroupIngressArgs(
            description="SSH",
            from_port=22, to_port=22,
            protocol="tcp", cidr_blocks=["0.0.0.0/0"],
        ),
        # FastAPI inference service
        aws.ec2.SecurityGroupIngressArgs(
            description="FastAPI",
            from_port=8000, to_port=8000,
            protocol="tcp", cidr_blocks=["0.0.0.0/0"],
        ),
        # MLflow tracking server
        aws.ec2.SecurityGroupIngressArgs(
            description="MLflow",
            from_port=5000, to_port=5000,
            protocol="tcp", cidr_blocks=["0.0.0.0/0"],
        ),
        # Prometheus
        aws.ec2.SecurityGroupIngressArgs(
            description="Prometheus",
            from_port=9090, to_port=9090,
            protocol="tcp", cidr_blocks=["0.0.0.0/0"],
        ),
        # Grafana
        aws.ec2.SecurityGroupIngressArgs(
            description="Grafana",
            from_port=3000, to_port=3000,
            protocol="tcp", cidr_blocks=["0.0.0.0/0"],
        ),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            description="Allow all outbound",
            from_port=0, to_port=0,
            protocol="-1", cidr_blocks=["0.0.0.0/0"],
        )
    ],
    tags={**TAGS, "Name": "modelserve-sg"},
)

# ══════════════════════════════════════════════════════════════════════════════
# KEY PAIR  (SSH_PUBLIC_KEY env var set before pulumi up)
# ══════════════════════════════════════════════════════════════════════════════

ssh_public_key = os.environ.get("SSH_PUBLIC_KEY", "")
key_pair = aws.ec2.KeyPair(
    "modelserve-keypair",
    key_name="modelserve-keypair",
    public_key=ssh_public_key,
    tags={**TAGS, "Name": "modelserve-keypair"},
)

# ══════════════════════════════════════════════════════════════════════════════
# IAM — EC2 Instance Role (S3 + ECR access)
# ══════════════════════════════════════════════════════════════════════════════

assume_role_policy = """{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}"""

iam_role = aws.iam.Role(
    "modelserve-ec2-role",
    name="modelserve-ec2-role",
    assume_role_policy=assume_role_policy,
    tags={**TAGS, "Name": "modelserve-ec2-role"},
)

# S3 full access (MLflow artifacts + Feast offline store)
aws.iam.RolePolicyAttachment(
    "modelserve-s3-policy",
    role=iam_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonS3FullAccess",
)

# ECR read access (pull Docker images)
aws.iam.RolePolicyAttachment(
    "modelserve-ecr-policy",
    role=iam_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
)

instance_profile = aws.iam.InstanceProfile(
    "modelserve-instance-profile",
    name="modelserve-instance-profile",
    role=iam_role.name,
    tags={**TAGS, "Name": "modelserve-instance-profile"},
)

# ══════════════════════════════════════════════════════════════════════════════
# S3 BUCKET — MLflow artifacts + Feast offline store
# ══════════════════════════════════════════════════════════════════════════════

s3_bucket = aws.s3.Bucket(
    "modelserve-artifacts",
    bucket_prefix="modelserve-artifacts-",
    force_destroy=True,
    versioning=aws.s3.BucketVersioningArgs(enabled=True),
    tags={**TAGS, "Name": "modelserve-artifacts"},
)

# Block all public access
aws.s3.BucketPublicAccessBlock(
    "modelserve-artifacts-pab",
    bucket=s3_bucket.id,
    block_public_acls=True,
    block_public_policy=True,
    ignore_public_acls=True,
    restrict_public_buckets=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# ECR REPOSITORY — Docker images
# ══════════════════════════════════════════════════════════════════════════════

ecr_repo = aws.ecr.Repository(
    "modelserve-api",
    name="modelserve-api",
    force_delete=True,  # IMPORTANT: allows pulumi destroy even with images present
    image_scanning_configuration=aws.ecr.RepositoryImageScanningConfigurationArgs(
        scan_on_push=True,
    ),
    tags={**TAGS, "Name": "modelserve-api"},
)

# ══════════════════════════════════════════════════════════════════════════════
# EC2 USER-DATA SCRIPT
# Bootstrap: install Docker, clone repo, train model, start full stack
# ══════════════════════════════════════════════════════════════════════════════

def make_user_data(s3_bucket_name: str) -> str:
    return textwrap.dedent(f"""\
        #!/bin/bash
        set -euxo pipefail
        exec > /var/log/userdata.log 2>&1

        # ── System packages ────────────────────────────────────────────────
        apt-get update -y
        apt-get install -y docker.io docker-compose-plugin git curl awscli python3-pip

        systemctl enable docker
        systemctl start docker
        usermod -aG docker ubuntu

        # ── Clone repository ───────────────────────────────────────────────
        git clone {GITHUB_REPO} /home/ubuntu/modelserve
        chown -R ubuntu:ubuntu /home/ubuntu/modelserve

        # ── Write .env for production ──────────────────────────────────────
        cat > /home/ubuntu/modelserve/.env << 'ENVEOF'
        POSTGRES_USER=mlflow
        POSTGRES_PASSWORD=mlflow
        POSTGRES_DB=mlflow
        MLFLOW_ARTIFACT_ROOT=s3://{s3_bucket_name}/mlflow-artifacts
        REDIS_URL=redis://redis:6379
        MLFLOW_TRACKING_URI=http://mlflow:5000
        MLFLOW_MODEL_NAME=fraud_detector
        FEAST_REPO_PATH=/app/feast_repo
        GF_SECURITY_ADMIN_USER=admin
        GF_SECURITY_ADMIN_PASSWORD=admin
        ENVEOF

        # ── Start supporting services (not API yet — no model registered) ──
        cd /home/ubuntu/modelserve
        docker compose up -d postgres redis mlflow

        # ── Wait for MLflow to be healthy ──────────────────────────────────
        echo "Waiting for MLflow..."
        for i in $(seq 1 60); do
            if curl -sf http://localhost:5000/health > /dev/null 2>&1; then
                echo "MLflow ready after ${{i}}x5s"
                break
            fi
            sleep 5
        done

        # ── Register fraud detection model in MLflow ───────────────────────
        docker compose run --rm \\
            -e MLFLOW_TRACKING_URI=http://mlflow:5000 \\
            api python training/train_from_parquet.py

        # ── Start full stack ───────────────────────────────────────────────
        docker compose up -d

        echo "ModelServe bootstrap complete!"
        """)


# Use pulumi.Output.apply to build user-data after s3_bucket name is resolved
user_data = s3_bucket.id.apply(make_user_data)

# ══════════════════════════════════════════════════════════════════════════════
# EC2 INSTANCE
# ══════════════════════════════════════════════════════════════════════════════

ec2_instance = aws.ec2.Instance(
    "modelserve-ec2",
    ami=ubuntu_ami.id,
    instance_type="t3.small",
    subnet_id=subnet.id,
    vpc_security_group_ids=[sg.id],
    key_name=key_pair.key_name,
    iam_instance_profile=instance_profile.name,
    user_data=user_data,
    user_data_replace_on_change=False,  # Don't replace instance on code change
    root_block_device=aws.ec2.InstanceRootBlockDeviceArgs(
        volume_size=20,
        volume_type="gp3",
        delete_on_termination=True,
    ),
    tags={**TAGS, "Name": "modelserve-server"},
)

# ══════════════════════════════════════════════════════════════════════════════
# ELASTIC IP — stable public address for TA demo
# ══════════════════════════════════════════════════════════════════════════════

eip = aws.ec2.Eip(
    "modelserve-eip",
    instance=ec2_instance.id,
    domain="vpc",
    tags={**TAGS, "Name": "modelserve-eip"},
)

# ══════════════════════════════════════════════════════════════════════════════
# STACK OUTPUTS — consumed by CI/CD pipeline and TA demo
# ══════════════════════════════════════════════════════════════════════════════

pulumi.export("instance_ip",        eip.public_ip)
pulumi.export("instance_id",        ec2_instance.id)
pulumi.export("ecr_repository_url", ecr_repo.repository_url)
pulumi.export("s3_bucket_name",     s3_bucket.id)
pulumi.export("vpc_id",             vpc.id)

pulumi.export("api_url",      eip.public_ip.apply(lambda ip: f"http://{ip}:8000"))
pulumi.export("mlflow_url",   eip.public_ip.apply(lambda ip: f"http://{ip}:5000"))
pulumi.export("grafana_url",  eip.public_ip.apply(lambda ip: f"http://{ip}:3000"))
pulumi.export("prometheus_url", eip.public_ip.apply(lambda ip: f"http://{ip}:9090"))

pulumi.export("ssh_command",  eip.public_ip.apply(
    lambda ip: f"ssh -i ~/.ssh/modelserve ubuntu@{ip}"
))
