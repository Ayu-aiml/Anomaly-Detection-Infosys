"use client";

import React, { useState, useRef } from 'react';
import axios from 'axios';
import { FaInfoCircle } from 'react-icons/fa';
import * as UTIF from 'utif';

const AnomalyDetection = () => {
    const [selectedModel, setSelectedModel] = useState('cnn');
    const [selectedImage, setSelectedImage] = useState(null);
    const [result, setResult] = useState(null);
    const [showDescription, setShowDescription] = useState(false);
    const [imagePreview, setImagePreview] = useState(null);
    const canvasRef = useRef(null);

    const modelDescriptions = {
        cnn: "CNN (Convolutional Neural Network) is a deep learning model often used for analyzing visual imagery.",
        mlp: "MLP (Multilayer Perceptron) is a class of feedforward artificial neural network used in pattern recognition.",
        fasterrcnn: "Faster R-CNN is an advanced object detection model that can detect anomalies by identifying objects in images."
    };

    const handleImageChange = (event) => {
        const file = event.target.files[0];
        setSelectedImage(file);

        if (file && (file.type === 'image/tiff' || file.type === 'image/tif')) {
            const reader = new FileReader();
            reader.onload = () => {
                const buffer = new Uint8Array(reader.result);
                const ifds = UTIF.decode(buffer);
                UTIF.decodeImage(buffer, ifds[0]); // Decoding the first page of the TIFF
                const rgba = UTIF.toRGBA8(ifds[0]);
                const canvas = document.createElement('canvas');
                canvas.width = ifds[0].width;
                canvas.height = ifds[0].height;
                const ctx = canvas.getContext('2d');
                const imageData = ctx.createImageData(canvas.width, canvas.height);
                imageData.data.set(rgba);
                ctx.putImageData(imageData, 0, 0);
                setImagePreview(canvas.toDataURL('image/png'));
            };
            reader.readAsArrayBuffer(file);
        } else {
            const reader = new FileReader();
            reader.onloadend = () => {
                setImagePreview(reader.result); // For other image types
            };
            if (file) {
                reader.readAsDataURL(file);
            }
        }
    };

    const handleModelChange = (event) => {
        setSelectedModel(event.target.value);
        setResult(null);
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
    };

    const drawBoxes = (boxes) => {
        const img = new Image();
        img.src = imagePreview;
        img.onload = () => {
            const canvas = canvasRef.current;
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);

            ctx.lineWidth = 2;
            ctx.strokeStyle = 'red';

            // Draw each box on the canvas
            boxes.forEach(box => {
                const [x1, y1, x2, y2] = box.map(coord => coord); // Scale the coordinates if needed
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            });
        };
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!selectedImage) {
            alert("Please select an image.");
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedImage);
        formData.append('model', selectedModel);

        try {
            setResult('Processing...');
            const response = await axios.post('http://localhost:5000/detect', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });

            if (selectedModel === 'fasterrcnn') {
                if (response.data.boxes && response.data.boxes.length > 0) {
                    drawBoxes(response.data.boxes);
                    setResult('Anomalies detected and highlighted.');
                } else {
                    setResult('No anomalies detected.');
                }
            } else {
                setResult(response.data.result || 'No result returned.');
            }
        } catch (error) {
            console.error('Error detecting anomaly', error);
            setResult('An error occurred while detecting anomaly.');
        }
    };

    return (
        <div className="anomaly-detection-container">
            <h1>Anomaly Detection</h1>
            <form onSubmit={handleSubmit} className="form-container">
                <div className="model-selection-container">
                    <label className="form-label">
                        Select Model:
                        <select value={selectedModel} onChange={handleModelChange} className="form-select">
                            <option value="cnn">CNN</option>
                            <option value="mlp">MLP</option>
                            <option value="fasterrcnn">Faster R-CNN</option>
                        </select>
                    </label>
                    <div
                        className="info-icon"
                        onMouseEnter={() => setShowDescription(true)}
                        onMouseLeave={() => setShowDescription(false)}
                    >
                        <FaInfoCircle size={24} />
                        {showDescription && (
                            <div className="description-tooltip">
                                {modelDescriptions[selectedModel]}
                            </div>
                        )}
                    </div>
                </div>
                <label className="form-label">
                    Upload Image:
                    <input type="file" accept=".jpg,.jpeg,.png,.tiff,.tif" onChange={handleImageChange} className="form-file-input" />
                </label>
                <button type="submit" className="submit-button">Detect Anomalies</button>
            </form>
            {result && <h2 className="result-text">{result}</h2>}
            {imagePreview && (
                <div className="image-container">
                    {selectedModel === 'fasterrcnn' ? (
                        <canvas ref={canvasRef} style={{ maxWidth: '100%', height: 'auto' }} />
                    ) : (
                        <img src={imagePreview} alt="Uploaded" style={{ maxWidth: '100%', height: 'auto' }} />
                    )}
                </div>
            )}
        </div>
    );
};

export default AnomalyDetection;
