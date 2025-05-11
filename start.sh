#!/bin/bash

# ✅ Gắn token nếu có (chỉ cần lần đầu hoặc dùng env)
if [ -n "$NGROK_AUTHTOKEN" ]; then
  echo "[+] Setting ngrok authtoken..."
  ngrok config add-authtoken "$NGROK_AUTHTOKEN"
fi

# ✅ Khởi động ngrok nền (forward port 8888)
echo "[+] Starting ngrok..."
ngrok http 8888 --log=stdout > /tmp/ngrok.log &


# ✅ Đợi ngrok sẵn sàng (tối đa 10s)
for i in {1..10}; do
  NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o 'https://[0-9a-z]*\.ngrok.io')
  if [[ $NGROK_URL != "" ]]; then
    echo "[+] Ngrok URL: $NGROK_URL"
    break
  fi
  sleep 1
done


# ✅ Chạy app
echo "[+] Starting FastAPI server..."
uvicorn server:app --host 0.0.0.0 --port 8888
