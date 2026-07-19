import cv2

# Camera configurations with multiple RTSP URL formats to try
CAMERA_CONFIGS = [
    {
        'name': 'Camera 1',
        'rtsp_urls': [
            'rtsp://admin:Eisa@1234@192.168.100.240:554/stream1',
            # 'rtsp://admin:Eisa@1234@192.168.100.239:554/h264',
            # 'rtsp://admin:Eisa@1234@192.168.100.239:554/1',
            # 'rtsp://admin:Eisa@1234@192.168.100.239:554',
            # 'rtsp://admin:Eisa@1234@192.168.100.239:80/stream',
        ],
        'ip': '192.168.100.240'
    },
    {
        'name': 'Camera 2',
        'rtsp_urls': [
            'rtsp://admin:Eisa@1234@192.168.100.239:554/stream1',
            # 'rtsp://admin:Eisa@1234@192.168.100.240:554/h264',
            # 'rtsp://admin:Eisa@1234@192.168.100.240:554/1',
            # 'rtsp://admin:Eisa@1234@192.168.100.240:554',
            # 'rtsp://admin:Eisa@1234@192.168.100.240:80/stream',
        ],
        'ip': '192.168.100.239'
    }
]

def main():
    # Initialize video captures for both cameras
    caps = []
    for config in CAMERA_CONFIGS:
        print(f"Connecting to {config['name']} at {config['ip']}...")
        
        # Try multiple RTSP URL formats
        cap = None
        for rtsp_url in config['rtsp_urls']:
            print(f"  Trying: {rtsp_url}")
            cap = cv2.VideoCapture(rtsp_url)
            
            # Set buffer size to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set preferred codec to H.264 to avoid HEVC issues
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            
            if cap.isOpened():
                # Try to read a frame to verify the connection works
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Successfully connected to {config['name']} at {config['ip']}")
                    caps.append((cap, config['name']))
                    break
                else:
                    cap.release()
                    print(f"  Connection established but no frame received")
            else:
                cap.release()
                print(f"  Failed to open connection")
        
        if cap is None or not any(c[0] == cap for c in caps):
            print(f"Failed to connect to {config['name']} after trying all URL formats")

    if not caps:
        print("No cameras connected. Exiting...")
        return

    print(f"Connected to {len(caps)} camera(s). Press 'q' to quit.")

    while True:
        frames = []
        for cap, name in caps:
            success, frame = cap.read()
            if success:
                # Add camera name to frame
                annotated = frame.copy()
                cv2.putText(annotated, name, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frames.append(annotated)
            else:
                print(f"Failed to read from {name}")

        if frames:
            # Display all camera frames side by side
            if len(frames) == 1:
                display_frame = frames[0]
            elif len(frames) == 2:
                # Stack horizontally
                display_frame = cv2.hconcat(frames)
            else:
                # For more than 2 cameras, create a grid
                display_frame = cv2.vconcat([cv2.hconcat(frames[i:i+2]) for i in range(0, len(frames), 2)])

            cv2.imshow("Live Camera Stream", display_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    for cap, name in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
