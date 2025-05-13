// // Runs when the extension is installed
// chrome.runtime.onInstalled.addListener(() => {
//   console.log("Extension installed!");
//   chrome.alarms.create("changeBackgroundAlarm", { periodInMinutes: 0.016 }); // Alarm triggers every second
// });
//
// // Alarm listener that executes a script to change background color
// chrome.alarms.onAlarm.addListener(async (alarm) => {
//   if (alarm.name === "changeBackgroundAlarm") {
//     const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
//     if (tab && tab.id) {
//       chrome.scripting.executeScript({
//         target: { tabId: tab.id },
//         function: changeBackgroundColor,
//       });
//     }
//   }
// });
//
// // Function injected into the webpage
// function changeBackgroundColor() {
//   const colors = ["lightblue", "lightgreen", "lightpink", "lightyellow", "lavender"];
//   document.body.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
// }

// background.js
chrome.runtime.onInstalled.addListener(() => {
  console.log("Real/AI Image Detector extension installed.");
});
