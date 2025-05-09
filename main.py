#Dewey Schoenfelder, Felix Bonilla, Patricia Bermeo, Jordan Berry
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import shutil
import numpy as np
from deepface import DeepFace
import cv2
import pickle
import requests  # For Telegram API
import time
import threading
import json
from picamera2 import Picamera2  # New library for Pi Camera
# pip install picamera2 DO THIS BE FOR RUNNING THE PROGRAM
# Telegram credentials
TELEGRAM_BOT_TOKEN = "7231732441:AAH3dKfLkKOFQJ0Bmc5anOA-on-T5Nxl1Fk"  # Replace with your bot token
TELEGRAM_CHAT_ID = None  # Will be set dynamically when the user starts a chat with the bot

# Global variable to store the last update ID
last_update_id = None

# Global variable for the camera frame (used in Tkinter)
global camera_frame
camera_frame = None

# Global variable for selected images
global selected_images
selected_images = []

# Global variable for face data
global face_data
face_data = []

# Global variable for picamera2 instance
global picam2
picam2 = None

# ----------------- Telegram & Face Recognition Functions -----------------

# Function to send a Telegram message with an image
def send_telegram_message(chat_id, image_path=None, message=None):
    try:
        if image_path:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {"photo": open(image_path, "rb")}
            payload = {
                "chat_id": chat_id,
                "caption": message
            }
            response = requests.post(url, files=files, data=payload)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message
            }
            response = requests.post(url, data=payload)

        if response.status_code == 200:
            print("Telegram message sent successfully!")
        else:
            print(f"Failed to send Telegram message. Error: {response.text}")
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

# Function to get updates from Telegram
def get_telegram_updates():
    global last_update_id
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {"offset": last_update_id + 1 if last_update_id else None, "timeout": 10}
    response = requests.get(url, params=params).json()

    if response.get("ok"):
        updates = response.get("result", [])
        if updates:
            last_update_id = updates[-1]["update_id"]
        return updates
    return []

# Function to retrieve the user's Chat ID
def get_chat_id_from_updates():
    updates = get_telegram_updates()
    for update in updates:
        if "message" in update:
            return update["message"]["chat"]["id"]  # Return the Chat ID
    return None

# Function to save the phone number and Chat ID
def save_phone_number_and_chat_id(phone_number, chat_id):
    data = {"phone_number": phone_number, "chat_id": chat_id}
    with open("phone_number_chat_id.json", "w") as f:
        json.dump(data, f)

# Function to load the phone number and Chat ID
def load_phone_number_and_chat_id():
    if os.path.exists("phone_number_chat_id.json"):
        with open("phone_number_chat_id.json", "r") as f:
            return json.load(f)
    return None

# Function to handle unknown face detection
def handle_unknown_face(image_path):
    # Load the phone number and Chat ID
    data = load_phone_number_and_chat_id()
    if not data or "chat_id" not in data:
        print("No Chat ID found. Please ensure the user has started a chat with the bot.")
        return

    chat_id = data["chat_id"]

    # Step 1: Send a Telegram message with the unknown face image
    send_telegram_message(chat_id, image_path, "🚨 DO YOU KNOW THE PERSON? Reply with /yes or /no.")

    # Step 2: Wait for the user's response
    response = wait_for_telegram_response()

    if response == "/yes":
        # Step 3: Ask for the person's name via Telegram
        send_telegram_message(chat_id, None, "PLEASE ENTER THE PERSON'S FIRST NAME AND LAST NAME.")
        person_name = wait_for_telegram_response()

        if person_name:
            # Step 4: Add the person to the system with three copies of the image
            add_person_to_system(image_path, person_name)
            # Refresh the camera feed to use the new data
            refresh_camera_feed()
    elif response == "/no":
        # Step 5: Send a second message with location info
        location = "Office Camera"  # Replace with actual location or use an API
        send_telegram_message(chat_id, None, f"THE INFO WILL BE SENT TO ALT + {location}.")
    else:
        print("Invalid response received.")

# Function to wait for a Telegram response
def wait_for_telegram_response():
    while True:
        updates = get_telegram_updates()
        for update in updates:
            if "message" in update and "text" in update["message"]:
                return update["message"]["text"].strip().lower()
        time.sleep(1)  # Wait for 1 second before checking again

# Function to add an unknown person to the system
def add_person_to_system(image_path, person_name):
    # Create a folder for the person
    folder_name = f"{person_name}"
    os.makedirs(folder_name, exist_ok=True)

    # Save three copies of the image in the folder
    for i in range(1, 4):  # Create 3 copies
        file_name = f"{folder_name}/image_{i}.jpg"
        shutil.copy(image_path, file_name)

    # Train the AI on the new images
    train_face_recognition_iteratively(folder_name, person_name)

    # Notify the user
    data = load_phone_number_and_chat_id()
    if data and "chat_id" in data:
        send_telegram_message(data["chat_id"], None, f"{person_name} has been added to the system.")

# Function to train the face recognition model
def train_face_recognition_iteratively(folder_name, person_name):
    # Load images from the folder
    face_encodings = []
    image_files = [f for f in os.listdir(folder_name) if f.endswith(".jpg")]

    # Iterative training loop
    max_iterations = 10  # Maximum number of iterations
    tolerance = 0.6  # Tolerance for face matching
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1} of training...")

        # Train on the current set of images
        for img_file in image_files:
            img_path = os.path.join(folder_name, img_file)
            try:
                # Load the image
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError(f"Could not load image: {img_path}")

                # Crop the face from the image
                cropped_face = crop_face(image)

                # Use DeepFace to extract facial embeddings from the cropped face
                embedding = np.array(DeepFace.represent(cropped_face, model_name="ArcFace", enforce_detection=False)[0]["embedding"])
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
                face_encodings.append(embedding)
                print(f"Embedding extracted for {img_file}")  # Debugging
            except Exception as e:
                messagebox.showwarning("Warning", f"Error processing {img_file}: {str(e)}")
                continue

        if not face_encodings:
            messagebox.showerror("Error", "No valid faces found in the folder.")
            return

        # Save the face encodings and the person's name using pickle
        data = {
            "name": person_name,
            "encodings": face_encodings
        }
        with open(f"{folder_name}/face_data.pkl", "wb") as f:
            pickle.dump(data, f)

        # Test the model on the same images
        confident = True
        for img_file in image_files:
            img_path = os.path.join(folder_name, img_file)
            try:
                # Load the image
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError(f"Could not load image: {img_path}")

                # Crop the face from the image
                cropped_face = crop_face(image)

                # Use DeepFace to extract facial embeddings from the cropped face
                face_embedding = np.array(DeepFace.represent(cropped_face, model_name="ArcFace", enforce_detection=False)[0]["embedding"])
                face_embedding = face_embedding / np.linalg.norm(face_embedding)  # Normalize

                # Check if the face matches any known faces
                match = False
                for encoding in face_encodings:
                    encoding = np.array(encoding)
                    encoding = encoding / np.linalg.norm(encoding)  # Normalize
                    distance = np.linalg.norm(encoding - face_embedding)
                    print(f"Distance to {person_name}: {distance}")  # Debugging
                    if distance > tolerance:  # If distance is too high, model is uncertain
                        confident = False
                        break

                if not confident:
                    break

            except Exception as e:
                print(f"Error processing face: {e}")

        if confident:
            print("Model is confident. Training complete.")
            break
        else:
            print("Model is uncertain. Adding more images and retraining...")
            # Add more images (e.g., prompt the user to add more images)
            messagebox.showinfo("Info", "The AI is uncertain. Please add more images of your face.")
            open_file_explorer()  # Allow the user to add more images
            image_files = [f for f in os.listdir(folder_name) if f.endswith(".jpg")]  # Update the list of images

    # Show a new window to notify the user that training is done
    training_done_window = tk.Toplevel(root)
    training_done_window.title("Training Complete")
    training_done_window.geometry("300x150")

    tk.Label(training_done_window, text="AI Training is Done!", font=("Arial", 14)).pack(pady=20)
    confirm_button = tk.Button(training_done_window, text="Confirm", font=("Arial", 14),
                               command=training_done_window.destroy)
    confirm_button.pack(pady=10)

# Function to crop a face from an image
def crop_face(image):
    # Detect faces in the image using MTCNN (more robust than Haar Cascade)
    try:
        detected_faces = DeepFace.extract_faces(image, detector_backend="mtcnn")
        if not detected_faces:
            raise ValueError("No faces detected in the image.")

        # Crop the first detected face
        face = detected_faces[0]
        cropped_face = face["face"]

        # Convert to RGB if the image has 4 channels (RGBA)
        if cropped_face.shape[2] == 4:
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGBA2RGB)

        # Resize the cropped face to a consistent size (e.g., 160x160)
        cropped_face = cv2.resize(cropped_face, (160, 160))

        return cropped_face
    except Exception as e:
        raise ValueError(f"Error detecting face: {e}")

# ----------------- GUI Functions -----------------

# Function to show the Add Photos screen
def show_add_photos_screen():
    # Clear the window and set up the Add Photos screen
    for widget in root.winfo_children():
        widget.destroy()

    # Add Photos screen title
    label = tk.Label(root, text="Add Photos (Only JPG)", font=("Arial", 24))
    label.pack(pady=20)

    global preview_frame
    preview_frame = tk.Frame(root)
    preview_frame.pack(pady=20)

    # Add button to open file explorer
    add_button = tk.Button(root, text="Add Pictures", font=("Arial", 18), width=20, height=2,
                           command=open_file_explorer)
    add_button.pack(expand=True, pady=10)

    # Take Picture button
    take_picture_button = tk.Button(root, text="Take Picture", font=("Arial", 18), width=20, height=2,
                                    command=take_picture)
    take_picture_button.pack(expand=True, pady=10)

    # Back button to return to the main menu
    back_button = tk.Button(root, text="Back to Main Menu", font=("Arial", 14), command=show_main_menu)
    back_button.pack(side="left", padx=20, pady=20)

# Function to open the file explorer
def open_file_explorer():
    global selected_images
    if len(selected_images) < 3:
        # Open file explorer to select JPG files
        file_path = filedialog.askopenfilename(filetypes=[("JPG Files", "*.jpg")])
        if file_path:
            selected_images.append(file_path)
            show_selected_images()

        if len(selected_images) == 3:
            ask_for_name_and_create_file()
    else:
        messagebox.showinfo("Info", "You have already selected 3 JPGs.")

# Function to show selected images
def show_selected_images():
    # Display previews of selected images in the window
    for widget in preview_frame.winfo_children():
        widget.destroy()

    for img_path in selected_images:
        img = Image.open(img_path)
        img.thumbnail((150, 150))
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(preview_frame, image=photo)
        label.image = photo  # Keep reference to avoid garbage collection
        label.pack(side="left", padx=10, pady=10)

# Function to ask for name and create a file
def ask_for_name_and_create_file():
    # Open a new window to enter first and last name
    def confirm_and_save():
        first_name = first_name_entry.get().strip()
        last_name = last_name_entry.get().strip()

        if not first_name or not last_name:
            messagebox.showerror("Error", "Please enter both first and last names.")
            return

        folder_name = f"{first_name}_{last_name}"
        os.makedirs(folder_name, exist_ok=True)

        for idx, img_path in enumerate(selected_images):
            file_name = f"{folder_name}/image_{idx + 1}.jpg"  # Save as JPG
            shutil.copy(img_path, file_name)

        # Train the AI on the images iteratively
        train_face_recognition_iteratively(folder_name, f"{first_name} {last_name}")

        name_window.destroy()
        selected_images.clear()
        show_main_menu()

        # Refresh the camera feed to use the new data
        refresh_camera_feed()

    # Create a new window for name input
    name_window = tk.Toplevel(root)
    name_window.title("Enter Name")
    name_window.geometry("400x200")

    tk.Label(name_window, text="First Name:", font=("Arial", 14)).pack(pady=5)
    first_name_entry = tk.Entry(name_window, font=("Arial", 14))
    first_name_entry.pack(pady=5)

    tk.Label(name_window, text="Last Name:", font=("Arial", 14)).pack(pady=5)
    last_name_entry = tk.Entry(name_window, font=("Arial", 14))
    last_name_entry.pack(pady=5)

    confirm_button = tk.Button(name_window, text="Confirm", font=("Arial", 14), command=confirm_and_save)
    confirm_button.pack(pady=20)

# Function to take pictures using the camera (using picamera2)
def take_picture():
    global selected_images, picam2
    # Initialize the camera via picamera2 if not already initialized
    if picam2 is None:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration())
        picam2.start()

    def capture_image():
        frame = picam2.capture_array()
        if frame is not None:
            # Convert to RGB if the image has 4 channels (RGBA)
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            image_path = f"captured_image_{len(selected_images) + 1}.jpg"
            cv2.imwrite(image_path, frame)
            selected_images.append(image_path)
            show_selected_images()

            if len(selected_images) == 3:
                # Do not stop picam2 here so it can be reused later
                ask_for_name_and_create_file()

    # Create a new window for the camera feed
    camera_window = tk.Toplevel(root)
    camera_window.title("Take Picture")

    # Frame to display the camera feed
    camera_frame = tk.Label(camera_window)
    camera_frame.pack()

    # Take Picture button
    take_picture_button = tk.Button(camera_window, text="Take Picture", font=("Arial", 14), command=capture_image)
    take_picture_button.pack(pady=10)

    def update_camera_feed():
        frame = picam2.capture_array()
        if frame is not None:
            # Convert to RGB if the image has 4 channels (RGBA)
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            img = Image.fromarray(frame)
            img.thumbnail((640, 480))
            photo = ImageTk.PhotoImage(img)
            camera_frame.config(image=photo)
            camera_frame.image = photo
        camera_frame.after(10, update_camera_feed)

    update_camera_feed()

# Function to show the Delete Photos screen
def show_delete_screen():
    # Clear the window and set up the Delete Photos screen
    for widget in root.winfo_children():
        widget.destroy()

    # Delete Photos screen title
    label = tk.Label(root, text="Delete Photos", font=("Arial", 24))
    label.pack(pady=20)

    # List of folders (created during training)
    folders = [f for f in os.listdir() if os.path.isdir(f) and os.path.exists(f"{f}/face_data.pkl")]

    # Listbox to display folder names
    global folder_listbox
    folder_listbox = tk.Listbox(root, font=("Arial", 14), selectmode=tk.SINGLE)
    for folder in folders:
        folder_listbox.insert(tk.END, folder)
    folder_listbox.pack(pady=20)

    # Confirm button to delete the selected folder
    confirm_button = tk.Button(root, text="Confirm", font=("Arial", 14), command=confirm_delete)
    confirm_button.pack(pady=10)

    # Back button to return to the main menu
    back_button = tk.Button(root, text="Back to Main Menu", font=("Arial", 14), command=show_main_menu)
    back_button.pack(side="bottom", pady=20)

# Function to confirm deletion of a folder
def confirm_delete():
    # Get the selected folder from the listbox
    selected_folder = folder_listbox.get(tk.ACTIVE)
    if not selected_folder:
        messagebox.showinfo("Info", "Please select a folder to delete.")
        return

    # Open a confirmation window
    confirm_window = tk.Toplevel(root)
    confirm_window.title("Confirm Delete")
    confirm_window.geometry("300x150")

    tk.Label(confirm_window, text=f"Are you sure you want to delete {selected_folder}?", font=("Arial", 14)).pack(pady=20)

    # Cancel button
    cancel_button = tk.Button(confirm_window, text="Cancel", font=("Arial", 14), command=confirm_window.destroy)
    cancel_button.pack(side="left", padx=20, pady=10)

    # Confirm button
    delete_button = tk.Button(confirm_window, text="Delete", font=("Arial", 14), command=lambda: delete_folder(selected_folder, confirm_window))
    delete_button.pack(side="right", padx=20, pady=10)

# Function to delete a folder
def delete_folder(folder_name, confirm_window):
    # Delete the folder and its contents
    try:
        shutil.rmtree(folder_name)
        messagebox.showinfo("Info", f"Folder {folder_name} deleted successfully.")
        confirm_window.destroy()
        show_delete_screen()  # Refresh the delete screen

        # Refresh the camera feed to use the updated data
        refresh_camera_feed()
    except Exception as e:
        messagebox.showerror("Error", f"Error deleting folder {folder_name}: {e}")

# Function to refresh the camera feed
def refresh_camera_feed():
    # Since we're using picamera2 globally, we don't need to release it here.
    # Simply call show_camera_screen() to refresh the display.
    show_camera_screen()

# Function to show the Camera screen (live face recognition)
def show_camera_screen():
    # Clear the window and set up the Camera screen
    for widget in root.winfo_children():
        widget.destroy()

    # Camera screen title
    label = tk.Label(root, text="Camera", font=("Arial", 24))
    label.pack(pady=20)

    # Frame to display the camera feed
    global camera_frame
    camera_frame = tk.Label(root)  # Initialize camera_frame
    camera_frame.pack()

    # Buttons at the bottom left
    button_frame = tk.Frame(root)
    button_frame.pack(side="bottom", anchor="sw", padx=20, pady=20)

    add_button = tk.Button(button_frame, text="Add/Shear", font=("Arial", 14), command=show_add_photos_screen)
    add_button.pack(side="left", padx=10)

    back_button = tk.Button(button_frame, text="Back", font=("Arial", 14), command=show_main_menu)
    back_button.pack(side="left", padx=10)

    delete_button = tk.Button(button_frame, text="Delete", font=("Arial", 14), command=show_delete_screen)
    delete_button.pack(side="left", padx=10)

    # Start the camera feed
    start_camera_feed()

# Function to start the camera feed and perform face recognition using picamera2
def start_camera_feed():
    # Load the saved face encodings
    global face_data, picam2
    face_data = []
    for folder in os.listdir():
        if os.path.isdir(folder) and os.path.exists(f"{folder}/face_data.pkl"):
            with open(f"{folder}/face_data.pkl", "rb") as f:
                data = pickle.load(f)
                face_data.append(data)
                print(f"Loaded face data for {data['name']}")  # Debugging

    # Initialize picamera2 if not already initialized
    if picam2 is None:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration())
        picam2.start()

    def update_camera_feed():
        frame = picam2.capture_array()
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                detected_faces = DeepFace.extract_faces(rgb_frame, detector_backend="mtcnn")
                for face in detected_faces:
                    x = face["facial_area"]["x"]
                    y = face["facial_area"]["y"]
                    w = face["facial_area"]["w"]
                    h = face["facial_area"]["h"]
                    face_region = frame[y:y+h, x:x+w]

                    # Convert to RGB if the image has 4 channels (RGBA)
                    if face_region.shape[2] == 4:
                        face_region = cv2.cvtColor(face_region, cv2.COLOR_RGBA2RGB)

                    try:
                        face_embedding = np.array(DeepFace.represent(face_region, model_name="ArcFace", enforce_detection=False)[0]["embedding"])
                        face_embedding = face_embedding / np.linalg.norm(face_embedding)  # Normalize

                        match = False
                        for data in face_data:
                            for encoding in data["encodings"]:
                                encoding = np.array(encoding)
                                encoding = encoding / np.linalg.norm(encoding)  # Normalize
                                distance = np.linalg.norm(encoding - face_embedding)
                                print(f"Distance to {data['name']}: {distance}")  # Debugging
                                if distance < 1.45:
                                    match = True
                                    name = data["name"]
                                    print(f"Match found: {name} (Distance: {distance})")  # Debugging
                                    break
                            if match:
                                break

                        color = (0, 255, 0) if match else (255, 0, 0)
                        label = name if match else "UNKNOWN"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        if not match:
                            unknown_face_path = "unknown_face.jpg"
                            cv2.imwrite(unknown_face_path, face_region)
                            handle_unknown_face(unknown_face_path)

                    except Exception as e:
                        print(f"Error processing face: {e}")

            except Exception as e:
                print(f"Error detecting faces: {e}")

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.thumbnail((640, 480))
            photo = ImageTk.PhotoImage(img)
            camera_frame.config(image=photo)
            camera_frame.image = photo
        camera_frame.after(10, update_camera_feed)

    update_camera_feed()

# Function to show the Enter Phone Number screen
def show_enter_phone_number_screen():
    # Clear the window and set up the Enter Phone Number screen
    for widget in root.winfo_children():
        widget.destroy()

    # Enter Phone Number screen title
    label = tk.Label(root, text="Enter Phone Number", font=("Arial", 24))
    label.pack(pady=20)

    # Entry field for phone number
    phone_entry = tk.Entry(root, font=("Arial", 14))
    phone_entry.pack(pady=20)

    # Confirm button
    def confirm_phone_number():
        phone_number = phone_entry.get().strip()
        if phone_number:
            # Ask the user to start a chat with the bot
            messagebox.showinfo("Info", "Please start a chat with the bot on Telegram to link your phone number.")
            chat_id = get_chat_id_from_updates()
            if chat_id:
                # Save the phone number and Chat ID
                save_phone_number_and_chat_id(phone_number, chat_id)
                messagebox.showinfo("Success", "Phone number and Chat ID saved successfully!")
                show_main_menu()  # Return to the main menu
            else:
                messagebox.showerror("Error", "No Chat ID found. Please start a chat with the bot.")
        else:
            messagebox.showerror("Error", "Please enter a valid phone number.")

    confirm_button = tk.Button(root, text="Confirm", font=("Arial", 14), command=confirm_phone_number)
    confirm_button.pack(pady=10)

    # Back button to return to the main menu
    back_button = tk.Button(root, text="Back to Main Menu", font=("Arial", 14), command=show_main_menu)
    back_button.pack(side="bottom", pady=20)

# Function to show the main menu
def show_main_menu():
    # Clear the window and set up the Main Menu screen
    for widget in root.winfo_children():
        widget.destroy()

    # Main Menu title
    label = tk.Label(root, text="Main Menu", font=("Arial", 24))
    label.pack(pady=20)

    # Create a frame for the buttons and place it in the center
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True)

    # Add buttons to the frame
    button1 = tk.Button(button_frame, text="Add/Shear Photos", font=("Arial", 18), width=20, height=2,
                        command=show_add_photos_screen)
    button1.pack(pady=10)

    button2 = tk.Button(button_frame, text="Delete Photos", font=("Arial", 18), width=20, height=2,
                        command=show_delete_screen)
    button2.pack(pady=10)

    button3 = tk.Button(button_frame, text="Main Camera", font=("Arial", 18), width=20, height=2,
                        command=show_camera_screen)
    button3.pack(pady=10)

    button4 = tk.Button(button_frame, text="Enter Phone Number", font=("Arial", 18), width=20, height=2,
                        command=show_enter_phone_number_screen)
    button4.pack(pady=10)

# ----------------- Main Program -----------------

# Create the main window
root = tk.Tk()
root.title("Face Recognition App")
root.geometry("900x900")

# Show the main menu initially
show_main_menu()

# Run the application
root.mainloop()