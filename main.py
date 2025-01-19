import streamlit as st
import cv2
import numpy as np
from keras import models
import random


@st.cache_resource
def load_model(file):
    mod = models.load_model(file)
    return mod


model = load_model("models/rps_v01_56ep_0.9641acc_0.1089loss.h5")

class_labels = {0: 'paper', 1: 'rock', 2: 'scissors'}

if "wins" not in st.session_state:
    st.session_state["wins"] = 0
if "losses" not in st.session_state:
    st.session_state["losses"] = 0
if "ties" not in st.session_state:
    st.session_state["ties"] = 0
if "total_guesses" not in st.session_state:
    st.session_state["total_guesses"] = 0
if "correct_guesses" not in st.session_state:
    st.session_state["correct_guesses"] = 0
if "failed_button_disable" not in st.session_state:
    st.session_state["failed_button_disable"] = False


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
    st.session_state["correct_guesses"] += 1
    st.session_state["failed_button_disable"] = False

    if player_choice == opponent_choice:
        st.session_state["ties"] += 1
        return ":grey[It's a tie!]"
    elif (
            (player_choice == "rock" and opponent_choice == "scissors") or
            (player_choice == "paper" and opponent_choice == "rock") or
            (player_choice == "scissors" and opponent_choice == "paper")
    ):
        st.session_state["wins"] += 1
        return ":violet[You win!] :sunglasses:"
    else:
        st.session_state["losses"] += 1
        return ":blue[You lose!] :sob:"


def calculate_accuracy():
    if st.session_state["total_guesses"] == 0:
        return 0.0
    return (st.session_state["correct_guesses"] / st.session_state["total_guesses"]) * 100


@st.fragment()
def show_result(res):
    wins, losses, ties = st.columns(3, vertical_alignment="center", border=True)
    wins.subheader("Wins: " + str(st.session_state["wins"]))
    losses.subheader("Losses: " + str(st.session_state["losses"]))
    ties.subheader("Ties: " + str(st.session_state["ties"]))

    st.divider()

    left, right = st.columns(2)

    with right:
        with st.expander("Prediction details"):
            st.image(cv2_img, channels="BGR", caption="Captured Image with ROI")
            st.write(f"Predicted: **{predicted_class}** with probability: **{probability:.2f}**")

    with left:
        failed_button = st.button("Mark guess as failed", disabled=st.session_state.get("failed_button_disable", True))
        if failed_button:
            st.session_state["correct_guesses"] -= 1
            if res == ":violet[You win!] :sunglasses:":
                st.session_state["wins"] -= 1
            elif res == ":blue[You lose!] :sob:":
                st.session_state["losses"] -= 1
            else:
                st.session_state["ties"] -= 1
            st.session_state["failed_button_disable"] = True
            st.rerun(scope="fragment")

        accuracy = calculate_accuracy()
        st.write(f"Accuracy: **{accuracy:.2f}%** of guesses were marked correct.")


st.title("Rock Paper Scissors 👊✌️✋")

roi_x, roi_y, roi_w, roi_h = 200, 50, 300, 300  # x=200, y=50, width=300, height=300

with st.sidebar:
    st.title("Tutorial 📙")
    st.write("1. When asked for camera permissions, select your webcam and agree")
    st.write("2. Position your hand in the center of the screen, as shown on the picture below:")
    st.image("media/example.jpg")
    st.write("3. After picking your shape, click the Take Picture button")
    st.write("4. If you want to play again, click the Clear Photo button")
    st.divider()
    st.write("Note: The prediction works best on simple, single-color backgrounds")

img_file_buffer = st.camera_input("Webcam preview 📷")

if img_file_buffer is not None:
    st.session_state["total_guesses"] += 1

    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    cv2.rectangle(cv2_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

    roi = cv2_img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    predicted_class, probability = predict_image(roi)

    random_shape = random_rps()

    match predicted_class:
        case "rock":
            pred_text = "rock 👊"
        case "paper":
            pred_text = "paper ✋"
        case "scissors":
            pred_text = "scissors ✌️️"
        case _:
            pred_text = "Error"

    match random_shape:
        case "rock":
            rand_text = "rock 👊"
        case "paper":
            rand_text = "paper ✋"
        case "scissors":
            rand_text = "scissors ✌️️"
        case _:
            rand_text = "Error"

    player_pick, random_pick = st.columns(2, vertical_alignment="center")
    player_pick.subheader("You picked:")
    player_pick.title(pred_text)
    random_pick.subheader("I picked:")
    random_pick.title(rand_text)
    st.divider()
    rps_res = rps_result(predicted_class, random_shape)
    result = st.columns([0.3, 0.4, 0.3])[1]
    result.title(rps_res)
    st.divider()

    show_result(rps_res)
