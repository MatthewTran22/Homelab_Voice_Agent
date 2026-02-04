FROM golang:1.24-alpine AS builder

WORKDIR /app

COPY go.mod ./
COPY agent/ ./agent/

RUN go build -o agent-binary ./agent/main.go

FROM alpine:latest

WORKDIR /app

COPY --from=builder /app/agent-binary .

CMD ["./agent-binary"]
