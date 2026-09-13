"""AWS 프리티어 S3 버킷 1회 프로비저닝 — CCTV 데모 이미지 정적 호스팅용.

실행 전 `aws configure`(또는 환경변수)로 자격증명이 설정돼 있어야 한다.
프리티어 범위(5GB, 요청 2만/월) 안에서 쓸 것을 가정 — 이 스크립트는 버킷을
만들고 "누구나 읽기(GetObject)만" 가능하도록 공개하는 정책을 붙인다(쓰기는
여전히 이 AWS 계정만 가능). 데모용 이미지 12장 + 모션 데모 10장 정도라
비용은 사실상 0원이어야 한다.

사용:
    python3 scripts/setup_s3_bucket.py --bucket fireway-cctv-demo --region ap-northeast-2
"""
from __future__ import annotations

import argparse
import json
import sys


def create_public_read_bucket(bucket: str, region: str) -> None:
    import boto3
    from botocore.exceptions import ClientError

    client = boto3.client("s3", region_name=region)

    try:
        if region == "us-east-1":
            client.create_bucket(Bucket=bucket)
        else:
            client.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        print(f"버킷 생성됨: {bucket} ({region})")
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code", "")
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"버킷 이미 존재: {bucket} (계속 진행)")
        else:
            raise

    # 데모용 공개 읽기 버킷이므로 계정 차원의 Block Public Access를 이 버킷에서만 해제.
    client.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": False,
            "IgnorePublicAcls": False,
            "BlockPublicPolicy": False,
            "RestrictPublicBuckets": False,
        },
    )

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PublicReadOnly",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{bucket}/*",
            }
        ],
    }
    client.put_bucket_policy(Bucket=bucket, Policy=json.dumps(policy))
    print("공개 읽기 정책(GetObject) 적용 완료 — 쓰기는 이 AWS 계정만 가능")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="S3 버킷 이름 (전역 고유해야 함)")
    parser.add_argument("--region", default="ap-northeast-2", help="AWS 리전 (기본: 서울)")
    args = parser.parse_args()

    try:
        create_public_read_bucket(args.bucket, args.region)
    except Exception as error:  # noqa: BLE001 — CLI 진입점, 사용자에게 원인 그대로 노출
        print(f"버킷 설정 실패: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
