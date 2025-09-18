import os
import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

# Configuration
MCAP_FILE = "mcap/spot_cam_correct/spot_cam_0.mcap"  # Change this to your MCAP file path
TOPIC_NAME = "/out/compressed"  # Change this to the image topic
OUTPUT_DIR = "extracted_images"
def extract_images_from_rosbag(mcap_file, topic_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Open the MCAP file
    storage_options = rosbag2_py.StorageOptions(uri=mcap_file, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions()
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Get available topics
    topic_types = reader.get_all_topics_and_types()
    topic_type_map = {topic.name: topic.type for topic in topic_types}

    if topic_name not in topic_type_map:
        print(f"Error: Topic '{topic_name}' not found in the bag file.")
        return

    if topic_type_map[topic_name] != "sensor_msgs/msg/CompressedImage":
        print(f"Error: Topic '{topic_name}' is not a CompressedImage topic.")
        return

    count = 0
    reader.seek(0)  # Reset to the beginning of the bag

    time = None
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        
        if topic == topic_name:
            try:
                # Deserialize CompressedImage message
                compressed_img_msg = deserialize_message(data, CompressedImage)
                if time == None:
                    time = compressed_img_msg.header.stamp
                elif time == compressed_img_msg.header.stamp:
                    continue
                else:
                    time = compressed_img_msg.header.stamp
                     
                print(f"Time: {compressed_img_msg.header.stamp}")
                # Decode the image
                np_arr = np.frombuffer(compressed_img_msg.data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if image is None:
                    print(f"Failed to decode image at index {count}")
                    continue

                # Save with zero-padded numbering (0001.jpg, 0002.jpg, etc.)
                filename = f"{count:04d}.png"
                filepath = os.path.join(output_dir, filename)

                cv2.imwrite(filepath, image)
                print(f"Saved: {filepath}")

                count += 1
            except Exception as e:
                print(f"Error processing image {count}: {e}")

    print(f"Extracted {count} images to {output_dir}")

# Run extraction
extract_images_from_rosbag(MCAP_FILE, TOPIC_NAME, OUTPUT_DIR)