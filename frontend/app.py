import streamlit as st
from api_client import query_assistant, detect_appliance, check_health

st.set_page_config(page_title="Kitchen Appliance Assistant", page_icon="🍳")
st.title("🍳 Kitchen Appliance Assistant")
st.caption("Ask a question about your microwave, oven, toaster, refrigerator, or coffee maker — "
           "answers are grounded in the appliance manuals with citations.")

if "history" not in st.session_state:
    st.session_state.history = []
if "appliance_hint" not in st.session_state:
    st.session_state.appliance_hint = None

with st.sidebar:
    st.subheader("Backend status")
    try:
        check_health()
        st.success("Backend is reachable")
    except Exception:
        st.error("Backend not reachable — is FastAPI running?")

    st.subheader("Vision component (Extended Track)")
    st.write("Upload a photo of the appliance to auto-detect it and ground answers to the right manual.")
    uploaded_image = st.file_uploader("Upload appliance photo", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded photo")
        if st.button("Detect appliance"):
            with st.spinner("Running YOLO detection..."):
                try:
                    result = detect_appliance(uploaded_image)
                    if result["appliance"]:
                        st.success(
                            f"Detected: **{result['appliance']}** "
                            f"(confidence: {result['confidence']:.2f})"
                        )
                        st.session_state.appliance_hint = result["manual_source"]
                    else:
                        st.warning("No recognizable appliance detected in the photo.")
                        st.session_state.appliance_hint = None
                except Exception as e:
                    st.error(f"Detection failed: {e}")

    if st.session_state.appliance_hint:
        st.info(f"Answers will be grounded to: {st.session_state.appliance_hint}")
        if st.button("Clear appliance filter"):
            st.session_state.appliance_hint = None

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn.get("sources"):
            st.caption("Sources: " + ", ".join(turn["sources"]))

question = st.chat_input("Ask a question about your appliance...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = query_assistant(question, appliance_hint=st.session_state.appliance_hint)
                st.write(result["answer"])
                if result["sources"]:
                    st.caption("Sources: " + ", ".join(result["sources"]))
                st.session_state.history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as e:
                error_msg = f"Sorry, I couldn't reach the assistant backend: {e}"
                st.error(error_msg)
                st.session_state.history.append({"role": "assistant", "content": error_msg})
