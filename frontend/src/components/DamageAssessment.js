import React, { useState } from "react";
import "./DamageAssessment.css";

function DamageAssessment() {
  const [disasterType, setDisasterType] = useState("");
  const [location, setLocation] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
    setResult(null);

    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleAssessClick = async () => {
    if (!selectedFile) {
      alert("Please upload an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("disaster_type", disasterType);
    formData.append("location", location);
    formData.append("timestamp", new Date().toISOString());

    setLoading(true);

    try {
      const response = await fetch("http://localhost:5000/analyze-damage", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to get assessment.");
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Assessment Error:", error);
      setResult({ error: "Error processing image. Please try again." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="content-card">
      <h2>Post-Disaster Damage Assessment</h2>
      <p>Upload an image (JPG, PNG) to assess the severity of damage.</p>

      {/* Context Inputs */}
      <div className="context-inputs">
        <select
          value={disasterType}
          onChange={(e) => setDisasterType(e.target.value)}
          className="context-select"
        >
          <option value="">Select Disaster Type</option>
          <option value="flood">Flood</option>
          <option value="fire">Fire</option>
          <option value="earthquake">Earthquake</option>
          <option value="cyclone">Cyclone</option>
        </select>

        <input
          type="text"
          placeholder="Enter Location (Area / City)"
          value={location}
          onChange={(e) => setLocation(e.target.value)}
          className="context-input"
        />
      </div>

      {/* Original Image Preview */}
      {previewUrl && (
        <img src={previewUrl} alt="Preview" className="preview-image" />
      )}

      {/* Upload Section */}
      <div className="file-upload-section">
        <input
          type="file"
          accept="image/jpeg, image/jpg"
          className="file-input"
          onChange={handleFileChange}
        />

        <button
          className="action-button"
          onClick={handleAssessClick}
          disabled={loading}
        >
          {loading ? "Analyzing Image..." : "Assess Damage"}
        </button>

        {loading && (
          <p className="loading-text">Processing image, please wait…</p>
        )}
      </div>

      {/* Results */}
      {result && (
        <div className="result-section">
          {/* Overlay Image */}
          {result.overlay_image && (
            <>
              <h4>AI Detected Damage Areas</h4>
              <img
                src={`data:image/png;base64,${result.overlay_image}`}
                alt="Damage Overlay"
                className="preview-image"
              />
            </>
          )}
          {result.explanation && (
  <div className="explanation-box">
    <h4>Why this assessment?</h4>
    <p>{result.explanation}</p>
  </div>
)}


          <h3>Assessment Result</h3>

          {result.error ? (
            <p className="error-message">{result.error}</p>
          ) : (
            <>
              <p>
                <strong>Disaster Type:</strong> {result.disaster_type}
              </p>
              <p>
                <strong>Location:</strong> {result.location}
              </p>
              <p>
                <strong>Time:</strong>{" "}
                {new Date(result.timestamp).toLocaleString()}
              </p>

              <ul className="result-list">
                <li>
                  No Damage: <strong>{result.building_no_damage}</strong>
                </li>
                <li>
                  Minor Damage:{" "}
                  <strong>{result.building_minor_damage}</strong>
                </li>
                <li>
                  Major Damage:{" "}
                  <strong>{result.building_major_damage}</strong>
                </li>
                <li>
                  Completely Destroyed:{" "}
                  <strong>
                    {result.building_complete_destruction}
                  </strong>
                </li>
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default DamageAssessment;
