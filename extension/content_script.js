// # Anna Scribner
// # Michael Gilbert
// # Muskan Gupta
// # Roger Tang

async function processImages() {
    let images = document.querySelectorAll('img');

    for (const img of images) {
        
        try {
            const response = await fetch("http://127.0.0.1:5000/serverTest", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ image_url: img.src })
            });

            const data = await response.json();
            console.log(img.src);
            console.log(data.result);
            console.log(data.accuracy);
            console.log("Server response:", data);


            // Here you can already put the watermark or process the image
        } catch (error) {
            console.error("Error contacting server:", error);
        }
    }
}

processImages();  // Call the async function
