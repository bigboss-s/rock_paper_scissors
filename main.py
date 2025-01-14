import streamlit as st
import cv2
import numpy as np
from keras import models
import random

model = models.load_model("models/rps_v01_56ep_0.9641acc_0.1089loss.h5")

class_labels = {0: 'paper', 1: 'rock', 2: 'scissors'}

if "wins" not in st.session_state:
    st.session_state["wins"] = 0
if "losses" not in st.session_state:
    st.session_state["losses"] = 0
if "ties" not in st.session_state:
    st.session_state["ties"] = 0


def preprocess_image(image, target_size=(150, 150)):
    img = cv2.resize(image, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(image):
    img = preprocess_image(image)
    prediction = model.predict(img, verbose=0)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    predicted_class_prob = prediction[predicted_class_index]
    return predicted_class_label, predicted_class_prob


def random_rps():
    shapes = ["scissors", "rock", "paper"]
    return random.choice(shapes)


def rps_result(player_choice, opponent_choice):
    if player_choice == opponent_choice:
        st.session_state["ties"] += 1
        return "It's a tie!"
    elif (
        (player_choice == "rock" and opponent_choice == "scissors") or
        (player_choice == "paper" and opponent_choice == "rock") or
        (player_choice == "scissors" and opponent_choice == "paper")
    ):
        st.balloons()
        st.session_state["wins"] += 1
        return "You win!"
    else:
        st.session_state["losses"] += 1
        return "You lose!"

st.title("Rock Paper Scissors 👊✌️✋")

roi_x, roi_y, roi_w, roi_h = 200, 50, 300, 300  # x=200, y=150, width=300, height=300

col1, col2 = st.columns([3, 1])

with st.expander("Tutorial", expanded=True, icon="📙"):
    st.write("1. When asked for camera permissions, select your webcam and agree")
    st.write("2. Position your hand in the center of the screen, as shown on the picture below:")
    st.image("media/example.jpg")
    st.write("3. After picking your shape, click the Take Picture button")
    st.write("4. If you want to play again, click the Clear Photo button")

img_file_buffer = st.camera_input("Webcam preview 📷")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    cv2.rectangle(cv2_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

    roi = cv2_img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    predicted_class, probability = predict_image(roi)

    random_shape = random_rps()

    match predicted_class:
        case "rock":
            pred_text = "You picked rock 👊"
        case "paper":
            pred_text = "You picked paper ✋"
        case "scissors":
            pred_text = "You picked scissors ✌️️"
        case _:
            pred_text = "Error"

    match random_shape:
        case "rock":
            rand_text = "I picked rock 👊"
        case "paper":
            rand_text = "I picked paper ✋"
        case "scissors":
            rand_text = "I picked scissors ✌️️"
        case _:
            rand_text = "Error"

    player_pick, random_pick, result = st.columns(3, vertical_alignment="center")
    player_pick.subheader(pred_text)
    random_pick.subheader(rand_text)
    result.subheader(rps_result(predicted_class, random_shape))

    wins, losses, ties = st.columns(3, vertical_alignment="center", border=True)
    wins.subheader("Wins: " + str(st.session_state["wins"]))
    losses.subheader("Losses: " + str(st.session_state["losses"]))
    ties.subheader("Ties: " + str(st.session_state["ties"]))

    with st.expander("Prediction details"):
        st.image(cv2_img, channels="BGR", caption="Captured Image with ROI")
        st.write(f"Predicted: **{predicted_class}** with probability: **{probability:.2f}**")
