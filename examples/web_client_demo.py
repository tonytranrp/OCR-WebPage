import os
import sys
import argparse

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network.web_server import WebServer

def main():
    parser = argparse.ArgumentParser(description='Web Camera Client Demo')
    parser.add_argument('--host', default='0.0.0.0',
                      help='Host address to bind to')
    parser.add_argument('--port', type=int, default=5000,
                      help='Web server port number (default: 5000)')
    parser.add_argument('--camera-port', type=int, default=8000,
                      help='Camera server port number (default: 8000)')
    parser.add_argument('--no-gpu', action='store_true',
                      help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    print("Starting Web Camera Client Demo")
    print(f"Web interface will be available at http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    print("Access this URL from any device on your local network to use the camera client")
    print("Press Ctrl+C to stop the server")
    
    server = WebServer(
        host=args.host,
        port=args.port,
        camera_server_port=args.camera_port,
        enable_gpu=not args.no_gpu
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.stop()

if __name__ == "__main__":
    main()