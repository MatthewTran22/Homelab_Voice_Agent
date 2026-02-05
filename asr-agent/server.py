#!/usr/bin/env python3
"""ASR gRPC Server - Transcribes audio files using Whisper."""

import os
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection
import whisper

import asr_pb2
import asr_pb2_grpc


class ASRServicer(asr_pb2_grpc.ASRServiceServicer):
    def __init__(self, model_name: str = "small"):
        print(f"Loading Whisper {model_name} model...")
        self.model = whisper.load_model(model_name)
        print("Model loaded successfully!")

    def Transcribe(self, request, context):
        file_path = request.file_path

        if not os.path.exists(file_path):
            return asr_pb2.TranscribeResponse(
                text="",
                success=False,
                error=f"File not found: {file_path}"
            )

        try:
            print(f"Transcribing: {file_path}")
            result = self.model.transcribe(file_path)
            text = result["text"].strip()
            print(f"Result: {text}")
            return asr_pb2.TranscribeResponse(
                text=text,
                success=True,
                error=""
            )
        except Exception as e:
            return asr_pb2.TranscribeResponse(
                text="",
                success=False,
                error=str(e)
            )


def serve(port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    asr_pb2_grpc.add_ASRServiceServicer_to_server(ASRServicer(), server)
    service_names = (
        asr_pb2.DESCRIPTOR.services_by_name["ASRService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)
    server.add_insecure_port(f"[::]:{port}")
    print(f"ASR gRPC server starting on port {port}...")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
