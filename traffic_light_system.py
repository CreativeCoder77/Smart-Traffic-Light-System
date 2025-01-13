import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time
import pickle
import threading
import warnings
import cv2
from ultralytics import YOLO
import os

warnings.filterwarnings("ignore", message="X does not have valid feature names")


class TrafficLightSystemWithYOLO:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Light Control System")
        self.root.geometry("600x1000")  # Width x Height
        self.root.resizable(False, False)

        # Load ML models
        with open("models/model_green.pkl", "rb") as f:
            self.model_green = pickle.load(f)
        with open("models/model_red.pkl", "rb") as f:
            self.model_red = pickle.load(f)

        # Load YOLO model
        self.yolo_model = YOLO("models/yolov8s-seg.pt")
        self.vehicle_class_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

        self.num_lanes = 2
        self.max_lanes = 4
        self.lanes_data = {}
        self.running = False
        self.image_paths = {}
        self.vehicle_counts = {}

        self.setup_gui()

    def setup_gui(self):
        # Style configuration
        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 10))
        style.configure("TButton", font=("Arial", 10))
        style.configure("TFrame", padding="5")
        style.configure("TLabelframe", font=("Arial", 12, "bold"), padding="10")

        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Lane configuration frame
        config_frame = ttk.LabelFrame(self.main_frame, text="Lane Configuration")
        config_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(config_frame, text="Number of Lanes (2-4):").pack(
            side=tk.LEFT, padx=5, pady=5
        )
        self.lane_count_var = tk.StringVar(value="2")
        lane_count_entry = ttk.Entry(
            config_frame, textvariable=self.lane_count_var, width=5
        )
        lane_count_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(config_frame, text="Update Lanes", command=self.update_lanes).pack(
            side=tk.LEFT, padx=5
        )

        # Lane images frame
        self.image_frame = ttk.LabelFrame(self.main_frame, text="Lane Images")
        self.image_frame.pack(fill=tk.X, padx=10, pady=5)

        # Detection frame
        self.detection_frame = ttk.LabelFrame(self.main_frame, text="Vehicle Detection")
        self.detection_frame.pack(fill=tk.X, padx=10, pady=5)

        # Traffic light frame
        self.lights_frame = ttk.LabelFrame(self.main_frame, text="Traffic Lights")
        self.lights_frame.pack(fill=tk.X, padx=10, pady=5)

        # Control buttons frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            self.button_frame, text="Detect Vehicles", command=self.detect_vehicles
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            self.button_frame, text="Start Simulation", command=self.start_simulation
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.button_frame, text="Stop", command=self.stop_simulation).pack(
            side=tk.LEFT, padx=5
        )

        # Detection text display
        self.detection_text = tk.Text(self.main_frame, height=10, wrap=tk.WORD)
        self.detection_text.pack(fill=tk.BOTH, padx=10, pady=5)

        # Initialize default lanes
        self.update_lanes()

    def update_lanes(self):
        try:
            new_num_lanes = int(self.lane_count_var.get())
            if not 2 <= new_num_lanes <= self.max_lanes:
                messagebox.showerror(
                    "Error", f"Number of lanes must be between 2 and {self.max_lanes}"
                )
                return

            self.num_lanes = new_num_lanes
            self.lanes_data.clear()
            self.image_paths.clear()
            self.vehicle_counts.clear()

            # Clear existing widgets
            for widget in self.image_frame.winfo_children():
                widget.destroy()
            for widget in self.detection_frame.winfo_children():
                widget.destroy()
            for widget in self.lights_frame.winfo_children():
                widget.destroy()

            # Create widgets for each lane
            for i in range(self.num_lanes):
                lane_label = f"Lane {i + 1}"

                # Image selection button
                ttk.Button(
                    self.image_frame,
                    text=f"Select Image for {lane_label}",
                    command=lambda x=i: self.select_image(x),
                ).grid(row=i, column=0, padx=5, pady=5, sticky=tk.W)

                # Image path label
                path_label = ttk.Label(self.image_frame, text="No image selected")
                path_label.grid(row=i, column=1, padx=5, pady=5, sticky=tk.W)

                # Detection labels
                count_label = ttk.Label(
                    self.detection_frame, text=f"{lane_label} - Vehicles: 0"
                )
                count_label.grid(row=i, column=0, padx=5, pady=5, sticky=tk.W)

                pred_label = ttk.Label(
                    self.detection_frame,
                    text=f"{lane_label} - Predicted: Green 0s, Red 0s",
                )
                pred_label.grid(row=i, column=1, padx=5, pady=5, sticky=tk.W)

                # Status label for each lane
                status_label = ttk.Label(
                    self.detection_frame, text=f"{lane_label} - Status: RED"
                )
                status_label.grid(row=i, column=2, padx=5, pady=5, sticky=tk.W)

                # Traffic light canvas
                canvas = tk.Canvas(self.lights_frame, width=80, height=180, bg="gray")
                canvas.grid(row=0, column=i, padx=10)
                self.create_traffic_light(canvas, lane_label)

                # Store lane data
                self.lanes_data[i] = {
                    "path_label": path_label,
                    "count_label": count_label,
                    "pred_label": pred_label,
                    "canvas": canvas,
                    "status_label": status_label,  # Fix for the missing key
                }

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of lanes")

    def select_image(self, lane_num):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.image_paths[lane_num] = file_path
            self.lanes_data[lane_num]["path_label"].config(
                text=os.path.basename(file_path)
            )

    def detect_vehicles(self):
        if not self.image_paths:
            messagebox.showerror("Error", "Please select images for all lanes first")
            return

        self.detection_text.delete(1.0, tk.END)
        self.detection_text.insert(tk.END, "Detecting vehicles...\n")

        for lane_num in range(self.num_lanes):
            if lane_num not in self.image_paths:
                messagebox.showerror(
                    "Error", f"Please select an image for Lane {lane_num+1}"
                )
                return

            image_path = self.image_paths[lane_num]
            image = cv2.imread(image_path)

            # Run YOLO detection
            results = self.yolo_model(image)
            vehicle_count = 0

            # Count vehicles
            for result in results:
                if result.boxes:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        if class_id in self.vehicle_class_ids:
                            vehicle_count += 1

            self.vehicle_counts[lane_num] = vehicle_count
            self.lanes_data[lane_num]["count_label"].config(
                text=f"Vehicles: {vehicle_count}"
            )

            # Display detection results
            self.detection_text.insert(
                tk.END, f"Lane {lane_num+1}: {vehicle_count} vehicles detected\n"
            )

            # Show detected image
            result_image = results[0].plot()
            cv2.imshow(f"Lane {lane_num+1} Detection", result_image)

        cv2.waitKey(1)  # Update windows

    def create_traffic_light(self, canvas, label):
        """
        Creates a traffic light with two circles (red and green)
        """
        canvas.create_text(45, 20, text=label)  # Label for the lane
        canvas.create_oval(20, 40, 65, 80, fill="darkred", tags="red")  # Red light
        canvas.create_oval(
            20, 90, 65, 130, fill="darkgreen", tags="green"
        )  # Green light

    def predict_times(self, cars):
        return (
            self.model_green.predict([[cars]])[0],
            self.model_red.predict([[cars]])[0],
        )

    def update_lights(self, lane_num, is_green):
        canvas = self.lanes_data[lane_num]["canvas"]
        if is_green:
            canvas.itemconfig("red", fill="darkred")
            canvas.itemconfig(
                "green", fill="lime"
            )  # Brighter green to make it more visible
        else:
            canvas.itemconfig("red", fill="red")
            canvas.itemconfig("green", fill="darkgreen")

    def start_simulation(self):
        if not self.vehicle_counts:
            messagebox.showerror("Error", "Please detect vehicles first")
            return

        if not self.running:
            self.running = True
            threading.Thread(target=self.run_simulation, daemon=True).start()

    def stop_simulation(self):
        self.running = False
        cv2.destroyAllWindows()

    def run_simulation(self):
        while self.running:
            try:
                # Sort lanes by vehicle count in descending order
                sorted_lanes = sorted(
                    range(self.num_lanes),
                    key=lambda lane: self.vehicle_counts[lane],
                    reverse=True,
                )

                lane_predictions = {}
                self.detection_text.delete(1.0, tk.END)

                # Prediction and logging for sorted lanes
                for lane_num in sorted_lanes:
                    cars = self.vehicle_counts[lane_num]
                    green_time, red_time = self.predict_times(cars)
                    lane_predictions[lane_num] = {
                        "cars": cars,
                        "green_time": green_time,
                        "red_time": red_time,
                    }

                    # Update prediction labels
                    self.lanes_data[lane_num]["pred_label"].config(
                        text=f"Predicted: Green {green_time:.1f}s, Red {red_time:.1f}s"
                    )

                    # Print predictions
                    self.detection_text.insert(
                        tk.END,
                        f"Lane {lane_num+1} ({cars} cars): Predicted Green = {green_time:.1f}s, Red = {red_time:.1f}s\n",
                    )

                # Simulation logic using sorted lanes
                for current_lane in sorted_lanes:
                    if not self.running:
                        return

                    # Update all traffic lights
                    for lane_num in range(self.num_lanes):
                        is_green = lane_num == current_lane
                        self.update_lights(lane_num, is_green)
                        self.lanes_data[lane_num]["status_label"].config(
                            text=f"Status: {'GREEN' if is_green else 'RED'}"
                        )

                    # Rest of the existing simulation logic remains the same
                    green_time = lane_predictions[current_lane]["green_time"]
                    start_time = time.time()

                    for i in range(int(green_time), -1, -1):
                        if not self.running:
                            return
                        for lane_num in range(self.num_lanes):
                            self.lanes_data[lane_num]["status_label"].config(
                                text=f"Status: {'GREEN' if lane_num == current_lane else 'RED'} ({i}s)"
                            )
                        time.sleep(1)

                    actual_time = time.time() - start_time
                    self.detection_text.insert(
                        tk.END,
                        f"Actual time for Lane {current_lane+1} Green: {actual_time:.1f}s\n",
                    )

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")
                self.running = False
                return


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficLightSystemWithYOLO(root)
    root.mainloop()
