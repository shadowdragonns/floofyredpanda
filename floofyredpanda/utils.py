import json
import socket


def requesttobyte(path, content_type="json", method="get", body=None, host="localhost"):
    method = method.upper()
    if not path.startswith("/"):
        path = "/" + path

    if content_type == "json":
        mime = "application/json"
    elif content_type == "form":
        mime = "application/x-www-form-urlencoded"
    else:
        mime = "text/plain"

    body = body or ""
    if isinstance(body, dict):
        import json
        body = json.dumps(body)

    if not isinstance(body, str):
        raise TypeError("Body must be a str or dict")

    body_bytes = body.encode("utf-8")
    content_length = len(body_bytes)

    headers = [
        f"{method} {path} HTTP/1.1",
        f"Host: {host}",
        f"User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
        f"Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language: en-GB,en;q=0.5",
        "Accept-Encoding: gzip, deflate, br",
        "Connection: close",
        "Upgrade-Insecure-Requests: 1",
        "Sec-Fetch-Dest: document",
        "Sec-Fetch-Mode: navigate",
        "Sec-Fetch-Site: none",
        "Sec-Fetch-User: ?1",
        f"Content-Type: {mime}",
        f"Content-Length: {content_length}",
        "",
        ""
    ]
    request_str = "\r\n".join(headers)
    full_request = request_str.encode("utf-8") + body_bytes
    return full_request


def request(host, port, endpoint):

 # Create a socket
 s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
 s.connect((host, port))

# Send HTTP GET request
 request = f"GET {endpoint} HTTP/1.1\r\nHost: {host}\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
 s.sendall(request.encode())

# Receive response
 response = b""
 while True:
     part = s.recv(4096)
     if not part:
         break
     response += part

 s.close()

 # Decode response
 response_text = response.decode('utf-8', errors='replace')  # be gentle with decoding

 # Split headers and body
 header_text, _, body = response_text.partition('\r\n\r\n')

 # Optional: print headers if curious
 # print(header_text)
 status_line = response_text.split('\r\n')[0]
 print("Status Line:", status_line)

 # Extract and print status code
 status_code = int(status_line.split()[1])
 print("Status Code:", status_code)
 # Parse JSON body
 try:
     data = json.loads(body)
     print("Parsed JSON:", data)
 except json.JSONDecodeError as e:
     print("Failed to parse JSON:", e)
     print("Raw body:", body)
     data = None
 return data , status_code