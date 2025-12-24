from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
import sys
import os
import base64
from tensorflow.keras.models import load_model
from joblib import load

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.connect_db import fetch_resource_data, update_resources, log_allocation
from custom_layer import SkipConnLayer, AttentionLayer, MyMeanIOU

app = Flask(__name__)
def analyze_with_confidence(image, mask, segmentation_output):
    """
    Analyze predictions with confidence scores to filter out uncertain classifications
    """
    height, width = mask.shape
    
    # Get confidence scores (max probability for each pixel)
    confidence_map = np.max(segmentation_output, axis=-1)
    
    print(f"\n📊 CONFIDENCE ANALYSIS:")
    print(f"   Min confidence: {confidence_map.min():.3f}")
    print(f"   Max confidence: {confidence_map.max():.3f}")
    print(f"   Mean confidence: {confidence_map.mean():.3f}")
    
    # Apply confidence threshold - only keep high-confidence predictions
    confidence_threshold = 0.60  # Adjust this (0.5-0.8)
    low_confidence_mask = confidence_map < confidence_threshold
    
    # Set low-confidence pixels to background
    filtered_mask = mask.copy()
    filtered_mask[low_confidence_mask] = 0
    
    low_conf_count = np.sum(low_confidence_mask)
    print(f"   🔻 Low confidence pixels removed: {low_conf_count} ({low_conf_count/(height*width)*100:.1f}%)")
    
    return filtered_mask, confidence_map
def remove_sky_mountains_with_texture(image, mask):
    """
    Remove sky/mountains using color AND texture analysis
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = image.shape[:2]
    
    removal_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 1. TOP REGION ONLY (30%)
    top_zone = int(height * 0.30)
    
    # 2. BRIGHTNESS (sky)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    bright_mask[top_zone:, :] = 0  # Only top region
    
    # 3. GREEN (mountains/vegetation)
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 200])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 4. BLUE (sky/mountains)
    lower_blue = np.array([90, 30, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 5. TEXTURE - buildings have edges, sky doesn't
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edge_density = cv2.dilate(edges, kernel, iterations=1)
    _, low_texture = cv2.threshold(edge_density, 30, 255, cv2.THRESH_BINARY_INV)
    
    # Combine: nature colors AND low texture
    nature_mask = cv2.bitwise_or(green_mask, blue_mask)
    nature_mask = cv2.bitwise_or(nature_mask, bright_mask)
    removal_mask = cv2.bitwise_and(nature_mask, low_texture)
    
    # PROTECT building classes (3,4,5,6)
    building_mask = ((mask >= 3) & (mask <= 6)).astype(np.uint8) * 255
    removal_mask[building_mask > 0] = 0
    
    # Clean up
    kernel = np.ones((7, 7), np.uint8)
    removal_mask = cv2.morphologyEx(removal_mask, cv2.MORPH_CLOSE, kernel)
    
    cleaned_mask = mask.copy()
    cleaned_mask[removal_mask > 0] = 0
    
    removed = np.sum(removal_mask > 0)
    print(f"🧹 Background removed: {removed} px ({removed/(height*width)*100:.1f}%)")
    
    return cleaned_mask
def validate_building_regions(mask, image):
    """
    Check each detected region - remove if it's clearly NOT a building
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    validated_mask = mask.copy()
    
    for class_id in [3, 4, 5, 6]:  # All building classes
        binary = (mask == class_id).astype(np.uint8)
        
        if np.sum(binary) == 0:
            continue
        
        # Find separate regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        for i in range(1, num_labels):
            component = (labels == i).astype(np.uint8)
            size = stats[i, cv2.CC_STAT_AREA]
            
            # Remove very small regions (noise)
            if size < 200:
                validated_mask[component > 0] = 0
                continue
            
            # Get region position
            y = stats[i, cv2.CC_STAT_TOP]
            x = stats[i, cv2.CC_STAT_LEFT]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            w = stats[i, cv2.CC_STAT_WIDTH]
            
            if h == 0 or w == 0:
                continue
            
            center_y = y + h // 2
            
            # If in upper 25% of image, check if it's nature
            if center_y < image.shape[0] * 0.25:
                region_hsv = hsv[y:y+h, x:x+w]
                region_mask = component[y:y+h, x:x+w]
                
                if region_mask.sum() == 0:
                    continue
                
                mean_hue = np.mean(region_hsv[:, :, 0][region_mask > 0])
                mean_sat = np.mean(region_hsv[:, :, 1][region_mask > 0])
                mean_val = np.mean(region_hsv[:, :, 2][region_mask > 0])
                
                # Green mountains
                if 35 < mean_hue < 90 and mean_sat > 40:
                    validated_mask[component > 0] = 0
                    print(f"   ❌ Removed green region in upper area (Class {class_id})")
                    continue
                
                # Sky (bright + low saturation)
                if mean_val > 180 and mean_sat < 50:
                    validated_mask[component > 0] = 0
                    print(f"   ❌ Removed bright region (likely sky) (Class {class_id})")
                    continue
    
    return validated_mask
CORS(app)

custom_objects = {
    "SkipConnLayer": SkipConnLayer,
    "AttentionLayer": AttentionLayer,
    "MyMeanIOU": MyMeanIOU
}

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

try:
    print("🔍 Loading trained models and scalers...")
    model = load_model("dqn_model.h5", custom_objects={"mse": mse})
    scaler_X = load("scaler_X.pkl")
    scaler_Y = load("scaler_Y.pkl")

    with tf.keras.utils.custom_object_scope(custom_objects):
        segmentation_model = load_model("model.h5", compile=False)

    print("✅ Models and scalers loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models or scalers: {e}")
    exit(1)

# ------------------------------
# 🔹 TARGETED SKY/MOUNTAIN REMOVAL ONLY
# ------------------------------
def remove_sky_and_mountains(image, mask):
    """
    Remove ONLY sky and mountains - keep ALL building classes untouched
    Focus on upper regions with nature colors
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    height, width = image.shape[:2]
    
    # Initialize removal mask
    removal_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 1. POSITION-BASED: Focus on upper 40% where mountains/sky typically are
    upper_region = np.zeros((height, width), dtype=np.uint8)
    upper_region[0:int(height * 0.40), :] = 255
    
    # 2. SKY DETECTION - Very bright areas
    _, bright_sky = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    
    # 3. GREEN VEGETATION/MOUNTAINS
    # Dark green mountains
    lower_green1 = np.array([35, 25, 25])
    upper_green1 = np.array([90, 255, 180])
    green_mask = cv2.inRange(hsv, lower_green1, upper_green1)
    
    # Bright green (vegetation)
    lower_green2 = np.array([35, 40, 80])
    upper_green2 = np.array([85, 255, 255])
    bright_green = cv2.inRange(hsv, lower_green2, upper_green2)
    
    vegetation_mask = cv2.bitwise_or(green_mask, bright_green)
    
    # 4. BLUE AREAS (mountains/sky)
    lower_blue = np.array([90, 20, 40])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 5. GRAY MOUNTAINS
    lower_gray = np.array([0, 0, 60])
    upper_gray = np.array([180, 25, 150])
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Combine nature features
    nature_mask = cv2.bitwise_or(vegetation_mask, blue_mask)
    nature_mask = cv2.bitwise_or(nature_mask, gray_mask)
    nature_mask = cv2.bitwise_or(nature_mask, bright_sky)
    
    # Apply to upper region primarily
    nature_in_upper = cv2.bitwise_and(nature_mask, upper_region)
    
    # Expand to connected regions
    kernel = np.ones((9, 9), np.uint8)
    nature_in_upper = cv2.morphologyEx(nature_in_upper, cv2.MORPH_CLOSE, kernel)
    nature_in_upper = cv2.dilate(nature_in_upper, kernel, iterations=1)
    
    # CRITICAL: Only remove from non-building classes
    # Classes 3,4,5,6 are buildings - NEVER touch them
    building_mask = ((mask == 3) | (mask == 4) | (mask == 5) | (mask == 6)).astype(np.uint8) * 255
    
    # Remove nature, but PROTECT buildings
    removal_mask = nature_in_upper.copy()
    removal_mask[building_mask > 0] = 0  # Never remove building pixels
    
    # Apply removal
    cleaned_mask = mask.copy()
    cleaned_mask[removal_mask > 0] = 0
    
    # Stats
    removed = np.sum(removal_mask > 0)
    protected = np.sum(building_mask > 0)
    print(f"🌲 Nature removed: {removed} pixels ({removed/(height*width)*100:.1f}%)")
    print(f"🏢 Buildings protected: {protected} pixels ({protected/(height*width)*100:.1f}%)")
    
    return cleaned_mask

def create_overlay(original, mask):
    """
    VIVID overlay colors with maximum contrast
    """
    color_map = {
        3: (0, 255, 0),        # No Damage - GREEN
        4: (255, 255, 0),      # Minor - YELLOW
        5: (255, 140, 0),      # Major - DARK ORANGE
        6: (220, 20, 60)       # Destroyed - CRIMSON RED
    }

    overlay = original.copy()
    
    print(f"🎨 Creating overlay...")
    unique, counts = np.unique(mask, return_counts=True)
    
    for class_id, color in color_map.items():
        binary_mask = (mask == class_id).astype(np.uint8) * 255
        
        if np.sum(binary_mask) > 0:
            # Dilate to make more visible
            kernel = np.ones((2, 2), np.uint8)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
            binary_mask = cv2.medianBlur(binary_mask, 3)
            
            pixels = np.sum(binary_mask > 0)
            class_names = {3: "No Damage", 4: "Minor", 5: "Major", 6: "Destroyed"}
            print(f"   {class_names.get(class_id, f'Class {class_id}')}: {pixels} pixels")
            
            overlay[binary_mask > 0] = color

    # Maximum visibility: 70% overlay
    blended = cv2.addWeighted(original, 0.3, overlay, 0.7, 0)
    return blended

def generate_explanation(results):
    """
    ACCURATE damage assessment based on destruction levels
    """
    no_damage = results["building_no_damage"]
    minor = results["building_minor_damage"]
    major = results["building_major_damage"]
    destroyed = results["building_complete_destruction"]
    
    total = no_damage + minor + major + destroyed
    damaged = minor + major + destroyed
    
    print(f"\n{'='*60}")
    print(f"📊 DAMAGE ASSESSMENT RESULTS:")
    print(f"{'='*60}")
    print(f"   ✅ No Damage:           {no_damage} structures")
    print(f"   🟡 Minor Damage:        {minor} structures")
    print(f"   🟠 Major Damage:        {major} structures")
    print(f"   🔴 Destroyed:           {destroyed} structures")
    print(f"   {'─'*60}")
    print(f"   📈 Total Buildings:     {total}")
    damaged_pct = (damaged/total*100) if total > 0 else 0
    print(f"   ⚠️  Total Damaged:       {damaged} ({damaged_pct:.0f}%)")
    print(f"{'='*60}\n")
    
    if total == 0:
        return "⚠️ No buildings detected. Please upload an image with visible structures."
    
    if damaged == 0:
        return f"✅ All {no_damage} structure(s) intact - no damage detected."
    
    # Percentages
    destroyed_pct = (destroyed / total * 100) if total > 0 else 0
    major_pct = (major / total * 100) if total > 0 else 0
    minor_pct = (minor / total * 100) if total > 0 else 0
    
    # DESTROYED BUILDINGS - Top Priority
    if destroyed > 0:
        if destroyed_pct >= 75:
            return (f"🚨 CATASTROPHIC DESTRUCTION: {destroyed}/{total} structures ({destroyed_pct:.0f}%) "
                    f"are completely destroyed. These buildings are reduced to rubble and debris - "
                    f"walls demolished, roofs collapsed, total structural failure. No structures remain standing. "
                    f"🚨 CRITICAL EMERGENCY: Immediate search-and-rescue, heavy equipment, and emergency shelters required.")
        
        elif destroyed_pct >= 50:
            return (f"🚨 MASSIVE DESTRUCTION: {destroyed}/{total} structures ({destroyed_pct:.0f}%) completely destroyed. "
                    f"Buildings have suffered total collapse with walls broken down and roofs caved in. "
                    f"Structures are uninhabitable piles of concrete and debris. "
                    f"Additional: {major} major damage, {minor} minor. "
                    f"🚨 URGENT: Deploy rescue teams, heavy machinery, and medical support immediately.")
        
        elif destroyed_pct >= 25:
            return (f"🔴 SEVERE DESTRUCTION: {destroyed}/{total} structures ({destroyed_pct:.0f}%) completely destroyed. "
                    f"These buildings show total structural collapse - demolished walls, fallen roofs, "
                    f"reduced to rubble. Buildings are beyond repair. "
                    f"Additional damage: {major} major (severe structural damage), {minor} minor (repairable). "
                    f"Comprehensive emergency response needed for all {damaged} affected structures.")
        
        else:
            # Even 1 destroyed building is critical
            destroyed_word = "structure" if destroyed == 1 else "structures"
            return (f"🔴 CRITICAL DAMAGE: {destroyed} {destroyed_word} completely destroyed with total structural collapse. "
                    f"Walls are demolished, roofs have fallen, and these buildings are reduced to debris and rubble. "
                    f"Total structural failure - buildings uninhabitable and beyond repair. "
                    f"Additional impact: {major} with major damage (severe cracks/partial collapse), "
                    f"{minor} with minor damage (repairable). "
                    f"Emergency response required for all {damaged} damaged structures.")
    
    # MAJOR DAMAGE (no destroyed buildings)
    if major > 0:
        if major_pct >= 60:
            return (f"🟠 SEVERE STRUCTURAL DAMAGE: {major}/{total} structures ({major_pct:.0f}%) with major damage. "
                    f"Buildings show severe structural compromise: deep cracks in load-bearing walls, partial roof collapse, "
                    f"foundation damage, or significant tilting. Buildings are unsafe for habitation. "
                    f"Additional: {minor} minor, {no_damage} intact. Structural engineers and safety inspections required.")
        
        elif major > minor:
            return (f"🟠 MAJOR STRUCTURAL DAMAGE DOMINANT: {major} structures with serious structural damage. "
                    f"Issues include: large wall cracks, partial wall failures, significant roof damage, foundation shifts. "
                    f"Buildings unsafe for entry without professional assessment. "
                    f"Additional: {minor} minor damage, {no_damage} undamaged. Total affected: {damaged}/{total}.")
    
    # MINOR DAMAGE ONLY
    if minor > 0:
        if minor_pct >= 60:
            return (f"🟡 WIDESPREAD MINOR DAMAGE: {minor}/{total} structures ({minor_pct:.0f}%) with minor damage. "
                    f"Damage is primarily cosmetic or non-structural: broken windows, small cracks in walls, "
                    f"damaged roof tiles, or minor surface damage. Buildings remain structurally sound and habitable. "
                    f"Standard repairs needed. Status: {major} major, {no_damage} intact.")
        else:
            return (f"🟡 MODERATE DAMAGE: {minor} structure(s) with minor damage. "
                    f"Issues are repairable: cosmetic cracks, broken windows, minor wall damage. "
                    f"Buildings remain safe and structurally sound. "
                    f"Total impact: {damaged}/{total} structures. Other: {major} major, {no_damage} intact.")
    
    return f"Assessment complete: {damaged}/{total} structures damaged."

@app.route("/analyze-damage", methods=["POST"])
def analyze_damage():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided."}), 400

        file = request.files["image"]
        disaster_type = request.form.get("disaster_type", "Unknown")
        location = request.form.get("location", "Unknown")
        timestamp = request.form.get("timestamp", "")

        print("\n" + "="*70)
        print(f"🌍 DISASTER DAMAGE ASSESSMENT")
        print(f"   Type: {disaster_type}")
        print(f"   Location: {location}")
        print(f"   Time: {timestamp}")
        print("="*70 + "\n")

        # Process image
        image_np = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"📐 Image dimensions: {image.shape}")
        
        image_resized = cv2.resize(image, (480, 360))
        image_normalized = image_resized / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)

        # Segmentation
        print("🤖 Running damage detection model...")
        segmentation_output = segmentation_model.predict(image_input, verbose=0)[0]
        predicted_mask = segmentation_output.argmax(axis=-1).astype(np.uint8)
        
        # Print raw model output
        print(f"\n🔍 RAW MODEL PREDICTIONS:")
        unique_raw, counts_raw = np.unique(predicted_mask, return_counts=True)
        class_names = {0: "Background", 3: "No Damage", 4: "Minor", 5: "Major", 6: "Destroyed"}
        for cls, cnt in zip(unique_raw, counts_raw):
            pct = cnt / (480*360) * 100
            print(f"   Class {cls} ({class_names.get(cls, 'Unknown')}): {cnt:6d} px ({pct:5.1f}%)")
        
        # STEP 1: Apply confidence filtering
        predicted_mask, confidence_map = analyze_with_confidence(image_resized, predicted_mask, segmentation_output)
        
        # STEP 2: Remove sky/mountains with texture analysis
        predicted_mask = remove_sky_mountains_with_texture(image_resized, predicted_mask)
        
        # STEP 3: Validate building regions
        predicted_mask = validate_building_regions(predicted_mask, image_resized)
        
        # Print cleaned stats
        unique_clean, counts_clean = np.unique(predicted_mask, return_counts=True)
        print(f"\n✨ FINAL PREDICTIONS (after cleaning):")
        for cls, cnt in zip(unique_clean, counts_clean):
            if cls > 0:
                pct = cnt / (480*360) * 100
                print(f"   Class {cls} ({class_names.get(cls, 'Unknown')}): {cnt:6d} px ({pct:5.1f}%)")
        
        # Create overlay
        overlay_image = create_overlay(image_resized, predicted_mask)
        overlay_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", overlay_bgr)
        overlay_base64 = base64.b64encode(buffer).decode("utf-8")

        building_classes = {
            "building_no_damage": 3,
            "building_minor_damage": 4,
            "building_major_damage": 5,
            "building_complete_destruction": 6
        }

        def count_buildings(mask, class_id, min_size=300):
            """
            Count with LOWER threshold to detect more buildings
            """
            binary_mask = (mask == class_id).astype(np.uint8)
            
            total_pixels = np.sum(binary_mask)
            if total_pixels == 0:
                return 0
            
            # Gentle morphology
            kernel = np.ones((3, 3), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            
            # Connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            count = 0
            sizes = []
            
            for i in range(1, num_labels):
                size = stats[i, cv2.CC_STAT_AREA]
                if size >= min_size:
                    count += 1
                    sizes.append(size)
            
            if sizes:
                print(f"   Class {class_id}: Found {count} buildings (sizes: {sorted(sizes, reverse=True)[:3]})")
            
            return count

        # Count buildings
        print(f"\n🏢 COUNTING BUILDINGS:")
        results = {
            label: int(count_buildings(predicted_mask, class_id))
            for label, class_id in building_classes.items()
        }
        
        # Generate explanation
        explanation = generate_explanation(results)
        
        print(f"\n💬 {explanation}")
        print("="*70 + "\n")

        return jsonify({
            **results,
            "overlay_image": overlay_base64,
            "explanation": explanation,
            "disaster_type": disaster_type,
            "location": location,
            "timestamp": timestamp
        })

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/allocate-resources", methods=["POST"])
def allocate_resources():
    try:
        data = request.json
        print(f"📥 Resource Allocation Request: {data}")

        required_keys = ["building_no_damage", "building_minor_damage", 
                        "building_major_damage", "building_total_destruction"]
        if not all(key in data for key in required_keys):
            return jsonify({"error": "Invalid input format. Missing required fields."}), 400

        no_damage = int(data["building_no_damage"])
        num_minor = int(data["building_minor_damage"])
        num_major = int(data["building_major_damage"])
        num_total = int(data["building_total_destruction"])

        damage_input = np.array([[0, num_minor, num_major, num_total]])
        damage_scaled = scaler_X.transform(damage_input)
        predicted_scaled = model.predict(damage_scaled)
        predicted_allocations = scaler_Y.inverse_transform(predicted_scaled)[0]
        predicted_allocations = np.maximum(predicted_allocations, 0).astype(int).tolist()

        resource_data = fetch_resource_data()
        if resource_data is None or resource_data.empty:
            return jsonify({"error": "Resource data unavailable in the database."}), 500

        resource_names = resource_data["resource_name"].tolist()

        total_damaged_buildings = num_minor + num_major + num_total
        minor_allocations, major_allocations, total_allocations = [], [], []

        if total_damaged_buildings > 0:
            for allocated_quantity in predicted_allocations:
                minor_share = (num_minor / total_damaged_buildings) * allocated_quantity if num_minor > 0 else 0
                major_share = (num_major / total_damaged_buildings) * allocated_quantity if num_major > 0 else 0
                total_share = allocated_quantity - (int(minor_share) + int(major_share))

                minor_allocations.append(int(minor_share))
                major_allocations.append(int(major_share))
                total_allocations.append(int(total_share))

        allocation_results = {
            "minor_damage": [{"resource_name": resource_names[i], "allocated_quantity": minor_allocations[i]} 
                           for i in range(len(resource_names)) if minor_allocations[i] > 0],
            "major_damage": [{"resource_name": resource_names[i], "allocated_quantity": major_allocations[i]} 
                           for i in range(len(resource_names)) if major_allocations[i] > 0],
            "total_destruction": [{"resource_name": resource_names[i], "allocated_quantity": total_allocations[i]} 
                                for i in range(len(resource_names)) if total_allocations[i] > 0]
        }

        for i, resource_name in enumerate(resource_names):
            allocated_quantity = minor_allocations[i] + major_allocations[i] + total_allocations[i]
            if allocated_quantity > 0:
                update_resources(resource_name, int(allocated_quantity))
                log_allocation(1, i + 1, int(allocated_quantity))

        updated_resources = fetch_resource_data().to_dict(orient="records")

        print("✅ Resource Allocation Complete!")
        return jsonify({
            "allocation_results": allocation_results,
            "updated_resources": updated_resources,
            "building_no_damage": no_damage,
            "building_minor_damage": num_minor,
            "building_major_damage": num_major,
            "building_complete_destruction": num_total,
            "disaster_type": data.get("disaster_type", "Unknown"),
            "location": data.get("location", "Unknown"),
            "timestamp": data.get("timestamp", "")
        })

    except Exception as e:
        print(f"❌ Resource Allocation Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Disaster Response API Server")
    print("="*50)
    print("   Listening on: http://127.0.0.1:5000/")
    print("   Endpoints:")
    print("     - POST /analyze-damage")
    print("     - POST /allocate-resources")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)