import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# ── Nutrition lookup (calories per 100g / standard serving) ──────────────────
FOOD_NUTRITION = {
    # (calories, protein_g, carbs_g, fat_g, fiber_g)
    "rice":        (130, 2.7, 28, 0.3, 0.4),
    "bread":       (265, 9,   49, 3.2, 2.7),
    "egg":         (155, 13,  1.1,11,  0),
    "milk":        (42,  3.4, 5,  1,   0),
    "chicken":     (165, 31,  0,  3.6, 0),
    "fish":        (136, 26,  0,  3,   0),
    "vegetables":  (35,  2,   7,  0.3, 2.5),
    "fruits":      (52,  0.6, 14, 0.2, 2.4),
    "dal":         (116, 9,   20, 0.4, 8),
    "roti":        (297, 11,  63, 1.2, 2),
    "idli":        (58,  2,   11, 0.4, 0.5),
    "dosa":        (168, 3.8, 22, 7,   1),
    "oats":        (389, 17,  66, 7,   10),
    "banana":      (89,  1.1, 23, 0.3, 2.6),
    "rice and curry": (200, 5, 40, 3, 1.5),
    "sambar rice": (180, 6,   36, 2,   3),
    "chapati":     (297, 11,  63, 1.2, 2),
    "poha":        (180, 4,   34, 4,   2),
    "upma":        (160, 4,   28, 4,   2),
    "default":     (200, 5,   30, 5,   2),
}

def parse_meal_nutrition(meal_text):
    """Extract nutrition from free-text meal description."""
    meal_text = meal_text.lower().strip()
    total = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    matched = False
    for food, nutrients in FOOD_NUTRITION.items():
        if food in meal_text:
            total += np.array(nutrients)
            matched = True
    if not matched:
        total += np.array(FOOD_NUTRITION["default"])
    return total  # (cal, protein, carbs, fat, fiber)


def compute_health_score(age, weight, height, gender, activity, total_cal, protein, carbs, fat, fiber):
    """Rule-based health score (0–100)."""
    score = 100.0

    # BMI penalty
    bmi = weight / ((height / 100) ** 2)
    if bmi < 18.5:
        score -= 15
    elif bmi > 30:
        score -= 20
    elif bmi > 25:
        score -= 10

    # Calorie balance
    bmr = (10 * weight + 6.25 * height - 5 * age + (5 if gender == "male" else -161))
    activity_multipliers = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725}
    tdee = bmr * activity_multipliers.get(activity, 1.375)
    cal_diff = abs(total_cal - tdee)
    if cal_diff > 500:
        score -= 15
    elif cal_diff > 300:
        score -= 8

    # Nutrition balance
    if protein < 40:
        score -= 10
    if fiber < 15:
        score -= 8
    if fat > 80:
        score -= 10

    # Age factor
    if age > 60:
        score -= 5
    elif age > 45:
        score -= 3

    return max(0, min(100, score))


def generate_lifespan_curve(base_score, age):
    """Generate projected health scores across future age milestones."""
    milestones = list(range(age, min(age + 41, 91), 5))
    scores = []
    for future_age in milestones:
        delta = future_age - age
        decay = delta * 0.4  # natural decline
        s = base_score - decay
        scores.append(round(max(0, min(100, s)), 1))
    return milestones, scores


def get_recommendations(score, bmi, protein, fiber, fat, total_cal):
    """Generate personalized health recommendations."""
    recs = []
    if score >= 70:
        recs.append("✅ Great health profile! Keep maintaining your balanced diet.")
    if bmi < 18.5:
        recs.append("⚠️ You are underweight. Increase calorie-dense nutritious foods like nuts, dairy, and whole grains.")
    elif bmi > 30:
        recs.append("⚠️ High BMI detected. Reduce refined carbs and sugary foods. Walk 30 mins daily.")
    elif bmi > 25:
        recs.append("💡 Slightly above ideal BMI. Light exercise and portion control will help.")
    if protein < 40:
        recs.append("🥩 Low protein intake. Add eggs, chicken, lentils, or paneer to your meals.")
    if fiber < 15:
        recs.append("🥦 Low fiber. Include more vegetables, fruits, and whole grains daily.")
    if fat > 80:
        recs.append("🧈 High fat intake. Switch to olive oil, reduce fried foods and full-fat dairy.")
    if total_cal > 3000:
        recs.append("🔥 Very high calorie intake. Consider smaller portion sizes and mindful eating.")
    if score < 50:
        recs.append("🚨 Critical: Consult a nutritionist or doctor for a personalized diet plan.")
        recs.append("💧 Drink at least 2.5L of water daily.")
        recs.append("😴 Ensure 7–8 hours of sleep. Poor sleep worsens metabolic health.")
    elif score < 60:
        recs.append("💧 Stay hydrated — aim for 2L of water per day.")
        recs.append("🏃 Add 20–30 minutes of moderate exercise 4 times a week.")
    return recs


# ── Train & save a simple Random Forest on synthetic data ──────────────────
def train_and_save():
    np.random.seed(42)
    n = 2000
    ages = np.random.randint(15, 80, n)
    weights = np.random.randint(40, 120, n)
    heights = np.random.randint(145, 195, n)
    genders = np.random.choice([0, 1], n)         # 0=female, 1=male
    activities = np.random.choice([0, 1, 2, 3], n) # sedentary..active
    cals = np.random.randint(1200, 3500, n).astype(float)
    proteins = np.random.randint(20, 120, n).astype(float)
    fibers = np.random.randint(5, 40, n).astype(float)
    fats = np.random.randint(20, 100, n).astype(float)

    scores = []
    activity_map = {0: "sedentary", 1: "light", 2: "moderate", 3: "active"}
    gender_map = {0: "female", 1: "male"}
    for i in range(n):
        bmi = weights[i] / ((heights[i] / 100) ** 2)
        s = compute_health_score(
            ages[i], weights[i], heights[i],
            gender_map[genders[i]], activity_map[activities[i]],
            cals[i], proteins[i], fibers[i], fats[i], fibers[i]
        )
        scores.append(s)

    X = np.column_stack([ages, weights, heights, genders, activities, cals, proteins, fibers, fats])
    y = np.array(scores)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print("✅ Model trained and saved to ml_model/model.pkl")


if __name__ == "__main__":
    train_and_save()
