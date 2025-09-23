import socket
import ssl
import threading

def start_floofy_server(port, handler):
    s = socket.socket()
    s.bind(("0.0.0.0", port))
    s.listen(5)

    print(f"Floofy server listening on port {port}")



    while True:
        conn, addr = s.accept()
        print(f"New friend from {addr[0]}")



        try:
            handler(conn, addr)
        except Exception as e:
            print(f"Handler error: {e}")
            conn.close()
class ParanoidPanda:
    def __init__(self, port, handler,cert,key):
        self.port = port
        self.handel = handler
        self.cert = cert
        self.key = key
    def start_paranoid_server(self):
        x = self.contextget()
        s = socket.socket()
        s.bind(("0.0.0.0",self.port))
        s.listen(5)
        while True:
         c, a = s.accept()
         print(f"checking if {a[0]} is a red panda or a panda")
         try:
          conn = x.wrap_socket(c, server_side=True)
          print("They are a red panda allowing connection")

         except Exception as e:
             print(f"There were a panda closing connection {e}")
             try:
              c.shutdown(socket.SHUT_RDWR)
             except:
                 pass
             c.close()
         try:
          if conn is not None:
              threading.Thread(target=self.handel, args=(conn,a,),daemon=True).start()
         except:
             pass

    def contextget(self):
        x = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        x.load_cert_chain(keyfile=self.key, certfile=self.cert)

        return x


