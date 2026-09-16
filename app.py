import numpy as np
import tensorflow as tf
import gradio as gr
import os

# =========================================================
# إعدادات المشروع
# =========================================================

MODEL_PATH = "ultimate_strong_cat_dog_model.keras"
IMG_SIZE = (299, 299)


# =========================================================
# التأكد من وجود النموذج
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"لم يتم العثور على النموذج:\n{MODEL_PATH}"
    )


# =========================================================
# تحميل النموذج
# =========================================================

model = tf.keras.models.load_model(MODEL_PATH)


# =========================================================
# دالة التنبؤ
# =========================================================

def predict_image(image):

    if image is None:
        return "⚠️ يرجى اختيار صورة أولاً."

    try:
        # تحويل الصورة إلى RGB
        image = image.convert("RGB")

        # تغيير الحجم إلى الحجم الذي تدرب عليه النموذج
        image = image.resize(IMG_SIZE)

        # تحويل الصورة إلى NumPy
        x = np.array(image, dtype=np.float32)

        # إضافة Batch Dimension
        x = np.expand_dims(x, axis=0)

        # =================================================
        # مهم:
        # لا نستخدم preprocess_input هنا.
        # النموذج يحتوي على preprocessing داخليًا.
        # =================================================

        prediction = float(
            model.predict(x, verbose=0)[0][0]
        )

        # احتمال DOG
        p_dog = prediction

        # احتمال CAT
        p_cat = 1.0 - prediction

        # تحديد الفئة
        if prediction >= 0.5:

            result = "🐶 DOG"
            confidence = p_dog * 100

        else:

            result = "🐱 CAT"
            confidence = p_cat * 100

        # نتيجة مرتبة
        return (
            f"{result}\n\n"
            f"درجة الثقة: {confidence:.2f}%\n\n"
            f"🐱 CAT: {p_cat * 100:.2f}%\n"
            f"🐶 DOG: {p_dog * 100:.2f}%"
        )

    except Exception as e:

        return (
            "❌ حدث خطأ أثناء تحليل الصورة:\n\n"
            f"{str(e)}"
        )


# =========================================================
# إنشاء الواجهة
# =========================================================

with gr.Blocks(
    title="Cat vs Dog Classifier"
) as demo:

    gr.Markdown(
        """
        # 🐱🐶 Cat vs Dog Classifier

        ### نظام تصنيف الصور باستخدام Xception

        ارفع صورة لقطة أو كلب، وسيقوم النموذج بتحليلها.
        """
    )

    image = gr.Image(
        type="pil",
        label="📷 اختر صورة قط أو كلب"
    )

    button = gr.Button(
        "🔎 تحليل الصورة"
    )

    result = gr.Textbox(
        label="النتيجة",
        lines=7
    )

    button.click(
        fn=predict_image,
        inputs=image,
        outputs=result
    )


# =========================================================
# تشغيل التطبيق
# =========================================================

if __name__ == "__main__":

    demo.launch()