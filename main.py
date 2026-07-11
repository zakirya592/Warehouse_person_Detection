import cv2
from ultralytics import YOLO
from alarm import Alarm

# Initialize alarm
alarm = Alarm()

# Load YOLO model
model = YOLO("best.pt")

# PPE class mapping
PPE_CLASSES = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "Machinery",
    9: "Vehicle"
}

# Video path
video_path = "video-from-rawpixel-id-19297010-sd.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        break
    
    # Run detection
    results = model(frame)
    
    # Track which violations are detected
    detected_violations = set()
    
    # Create annotated frame
    annotated = frame.copy()
    
    # Process results
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = PPE_CLASSES[class_id]
            
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Define colors
            if label in ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]:
                color = (0, 0, 255)  # Red for violations
            else:
                color = (0, 255, 0)  # Green for compliant PPE
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            cv2.putText(annotated, f"{label} {confidence:.2f}", 
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            
            # Check for violations
            if label in ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]:
                detected_violations.add(label)
    
    # Play alarm only if ALL THREE violations are detected
    if len(detected_violations) == 3:
        alarm.play()
        print("ALARM: All PPE violations detected - NO-Hardhat, NO-Mask, NO-Safety Vest")
    else:
        alarm.stop()
        if detected_violations:
            print(f"Partial violations: {detected_violations} - Alarm not triggered")
    
    # Display annotated frame
    cv2.imshow("PPE Detection", annotated)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
alarm.stop()
