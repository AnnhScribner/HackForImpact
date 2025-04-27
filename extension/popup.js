// # Anna Scribner
// # Michael Gilbert
// # Muskan Gupta
// # Roger Tang

document.getElementById('changeColor').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    function: () => {
      document.body.style.backgroundColor = "lightblue";
    },
  });
});
