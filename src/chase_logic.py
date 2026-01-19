def predict_chasing_score(target, venue_avg, chasing_strength):
    expected = venue_avg + (chasing_strength - 0.5) * 20

    if target > venue_avg + 15:
        expected -= 10
    elif target < venue_avg - 10:
        expected += 10

    return int(min(expected, target))


def chase_success_probability(target, venue_avg, chasing_strength):
    prob = 0.5

    if target < venue_avg - 10:
        prob += 0.2
    elif target > venue_avg + 15:
        prob -= 0.25

    prob += (chasing_strength - 0.5) * 0.3

    return min(max(prob, 0.05), 0.95)
