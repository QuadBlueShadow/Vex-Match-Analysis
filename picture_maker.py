import cv2
import yt_dlp
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

video_path = "Videos/MechaR-16-1.mkv"

def get_youtube():
    video_path = "https://www.youtube.com/live/JmpAWqVQmWs?si=shA7H6bgVyPebK0z"

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_path, download=False)
        video_url = info["url"]

    return video_url

video_path = "https://play.boxcast.com/p/xg5f6cth5ddcmjsvzpsl/v/1080p-byteranges.m3u8?Expires=1780185600&Signature=fovhaj8AP0l0FEbf~QW6OSyGfjl8D56ep7ofkHthT22a--pYubD0TVC3PSTXCGQp8mIYOHqYCyTZymwEDGIo0umq4Tn-4xqdc6UQP2vXj0QOzpTulorMQQag5jcVmfssv--iqPDNhQ3o-e~hvjDHGuK43j2425vWownPXUls~5f25ehBs3BY5wqecViOQUqx-7ALrr2RO8YmsbfSwsAksICbF6N3UWv9D1-EQEsi1DUxv188LhR2T2T9Hi2Adn6uJUG3Zv34k03dY9NMjuvUNqMiqgitnswXU-LtFzOPxJ4m1XjGlWZ5CRPLCj3K8uI5GqQl2tYZLAKVKa0Y9Mg2cw__&Key-Pair-Id=APKAJ7GUCBQUK6NTWZCA"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Could not open video file")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}")
print(f"Frames: {frame_count}")
print(f"Resolution: {width}x{height}")

fps = 60
delay = int(1000 / fps) if fps > 0 else 33  # milliseconds

cap.set(cv2.CAP_PROP_POS_FRAMES, 36_000)

index = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    #cv2.imshow("Video", frame)

    if index % 500 == 0:
        # match = frame[0:50, 0:450]
        # #middle = gray_scale(frame[300:800, 700:1200])
        # time = frame[1039:1280, 1670:1920]

        # text = pytesseract.image_to_string(match).strip()
        # #text += ", " + pytesseract.image_to_string(middle).strip()
        # text += ", " + pytesseract.image_to_string(time).strip()
        
        print(index)
        cv2.imwrite(f"Pictures/pic_{index}.png", frame)
    index += 1

cap.release()
cv2.destroyAllWindows()