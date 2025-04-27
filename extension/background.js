// # Anna Scribner
// # Michael Gilbert
// # Muskan Gupta
// # Roger Tang

chrome.runtime.onInstalled.addListener(() => {
  console.log("Extension installed!");

  // Set up the alarm to fire every 1 second
  chrome.alarms.create("changeBackgroundAlarm", { periodInMinutes: 0.016 });
});

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name === "changeBackgroundAlarm") {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (tab && tab.id) {
      chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: changeBackgroundColor
      });
    }
  }
});

// This function gets injected into the page
function changeBackgroundColor() {
  const colors = ["lightblue", "lightgreen", "lightpink", "lightyellow", "lavender"];
  document.body.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
}
