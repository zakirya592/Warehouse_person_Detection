import cv2
import os
from datetime import datetime, timedelta


class ScreenshotManager:
    def __init__(self, screenshots_dir="screenshots", reset_time_seconds=30):
        self.screenshots_dir = screenshots_dir
        self.reset_time_seconds = reset_time_seconds  # Reset photographed persons after this many seconds
        self.photographed_persons = []  # List of dicts with person info and timestamp
        
        # Create screenshots directory
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)
    
    def _clean_old_photographed_persons(self):
        """Remove photographed persons that are older than reset_time_seconds"""
        current_time = datetime.now()
        self.photographed_persons = [
            person for person in self.photographed_persons
            if (current_time - person['timestamp']).total_seconds() < self.reset_time_seconds
        ]
    
    def take_screenshot(self, frame, violating_persons):
        """
        Take a full screenshot with only violating persons highlighted in red.
        
        Args:
            frame: The original video frame
            violating_persons: List of dictionaries containing violation info (label, x1, y1, x2, y2, confidence)
        
        Returns:
            str: Path to the saved screenshot, or None if no new persons to photograph
        """
        # Clean old photographed persons (allow re-photographing after reset_time_seconds)
        self._clean_old_photographed_persons()
        
        # Check if there are any new persons to photograph
        new_persons_to_photograph = []
        
        for person_info in violating_persons:
            # Check if this person was already photographed (similar location)
            is_new_person = True
            for photographed in self.photographed_persons:
                # Check if boxes overlap significantly (same person)
                overlap = (min(person_info['x2'], photographed['x2']) - max(person_info['x1'], photographed['x1'])) * \
                         (min(person_info['y2'], photographed['y2']) - max(person_info['y1'], photographed['y1']))
                if overlap > 0:  # If there's any overlap, consider it the same person
                    is_new_person = False
                    break
            
            if is_new_person:
                new_persons_to_photograph.append(person_info)
        
        if not new_persons_to_photograph:
            return None
        
        # Create a copy of the frame for the screenshot
        screenshot_frame = frame.copy()
        
        # Draw red boxes and labels only for violating persons
        for person_info in new_persons_to_photograph:
            cv2.rectangle(screenshot_frame, (person_info['x1'], person_info['y1']), 
                        (person_info['x2'], person_info['y2']), (0, 0, 255), 2)
            cv2.putText(screenshot_frame, f"{person_info['label']} {person_info['confidence']:.2f}", 
                       (person_info['x1'], person_info['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 0, 255), 2)
        
        # Save screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        violations_str = "_".join([p['label'] for p in new_persons_to_photograph])
        screenshot_path = os.path.join(self.screenshots_dir, f"violation_{violations_str}_{timestamp}.jpg")
        cv2.imwrite(screenshot_path, screenshot_frame)
        
        # Mark these persons as photographed with current timestamp
        current_time = datetime.now()
        for person_info in new_persons_to_photograph:
            person_with_timestamp = person_info.copy()
            person_with_timestamp['timestamp'] = current_time
            self.photographed_persons.append(person_with_timestamp)
        
        return screenshot_path
