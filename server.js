const express = require('express');
const multer = require('multer');
const path = require('path');
const { spawn } = require('child_process');

const app = express();
const port = process.env.PORT || 7000;

app.use(express.static('public'));

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'public/uploads/')
    },
    filename: function (req, file, cb) {
        cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname))
    }
});

const upload = multer({
    storage: storage,
    // Add file size limit for security (e.g., 5MB)
    limits: { fileSize: 5 * 1024 * 1024 },
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post('/predict', upload.single('xrayImage'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded.' });
    }

    const imagePath = req.file.path;
    const pythonProcess = spawn('python', ['python_inference/inference.py', imagePath]);

    let pythonError = '';

    // **CRITICAL CHANGE**: Listen for errors from the Python script
    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
        pythonError += data.toString();
    });

    pythonProcess.on('close', (code) => {
        // If there was an error message or the script exited with an error code
        if (pythonError || code !== 0) {
            console.log(`Python script exited with code ${code}`);
            const errorDetails = pythonError || 'The script encountered an unknown error.';
            return res.status(500).json({
                error: 'Failed to process the image.',
                details: errorDetails
            });
        }

        // If successful
        const originalFilePath = req.file.path.replace(/\\/g, "/").substring('public/'.length);
        const processedFileName = path.basename(req.file.path);
        const processedFilePath = `processed/${processedFileName}`;

        res.json({
            original: originalFilePath,
            processed: processedFilePath
        });
    });
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});

