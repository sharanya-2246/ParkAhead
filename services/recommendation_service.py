from datetime import datetime, timedelta

class RecommendationService:
    def __init__(self):
        pass
    
    def recommend(self, predictions, user_input):
        """
        predictions: list of prediction dicts for each facility
        user_input: dict with event_time, eta, preferences etc
        """
        event_time = user_input.get('event_time')
        eta = user_input.get('eta')
        safety_margin_min = user_input.get('safety_margin_min', 10)
        budget = user_input.get('budget')
        
        scored_options = []
        
        for pred in predictions:
            facility_id = pred['facility_id']
            capacity = pred.get('capacity')
            avail = pred.get('predicted_available_spaces')
            occ_rate = pred.get('predicted_occupancy_rate')
            
            travel_time = pred.get('travel_time_min', 15)
            search_time = pred.get('search_time_min', 5)
            walking_time = pred.get('walking_time_min', 3)
            parking_fee = pred.get('parking_fee', 0)
            
            total_to_venue = travel_time + search_time + walking_time
            venue_arrival = eta + timedelta(minutes=total_to_venue)
            time_remaining = (event_time - venue_arrival).total_seconds() / 60 if event_time and venue_arrival else None
            
            # Scoring
            score = 100
            reasons = []
            
            # Availability factor
            if avail is None:
                score -= 40
                reasons.append("Availability uncertain (insufficient data)")
            elif avail <= 0:
                score -= 60
                reasons.append("Predicted to be full or very congested")
            elif avail < 10:
                score -= 30
                reasons.append(f"Low predicted availability ({avail} spaces)")
            elif avail >= 20:
                score += 10
                reasons.append(f"Good predicted availability ({avail} spaces)")
            else:
                reasons.append(f"Moderate predicted availability ({avail} spaces)")
            
            # Time factor
            if time_remaining is not None:
                if time_remaining < 0:
                    score -= 80
                    reasons.append(f"May arrive {abs(int(time_remaining))} min late for event")
                elif time_remaining < safety_margin_min:
                    score -= 40
                    reasons.append(f"Tight timing - only {int(time_remaining)} min buffer")
                elif time_remaining >= 30:
                    score += 10
                    reasons.append(f"Comfortable {int(time_remaining)} min buffer before event")
            
            # Travel/walk factors
            if total_to_venue > 30:
                score -= 10
                reasons.append("Longer total travel+search+walk time")
            elif total_to_venue < 15:
                score += 5
            
            # Budget
            if budget is not None and parking_fee > budget:
                score -= 20
                reasons.append(f"Exceeds budget (₹{parking_fee})")
            
            scored_options.append({
                **pred,
                'travel_time_min': travel_time,
                'search_time_min': search_time,
                'walking_time_min': walking_time,
                'parking_fee': parking_fee,
                'total_to_venue_min': total_to_venue,
                'venue_arrival_time': venue_arrival,
                'time_remaining_min': int(time_remaining) if time_remaining is not None else None,
                'score': score,
                'reasons': reasons
            })
        
        # Sort by score descending
        scored_options.sort(key=lambda x: x['score'], reverse=True)
        
        if scored_options:
            best = scored_options[0]
            best['is_recommended'] = True
            best['recommendation_reason'] = self._generate_reason(best)
        
        return scored_options
    
    def _generate_reason(self, option):
        reasons = []
        if option.get('predicted_available_spaces') is not None:
            reasons.append(f"Predicted {option['predicted_available_spaces']} spaces available at arrival")
        reasons.append(f"Est. venue arrival in {option['total_to_venue_min']} min total (travel+search+walk)")
        if option.get('time_remaining_min') is not None:
            if option['time_remaining_min'] >= 0:
                reasons.append(f"{option['time_remaining_min']} min buffer before event")
            else:
                reasons.append(f"WARNING: May arrive {abs(option['time_remaining_min'])} min late")
        if option.get('parking_fee') is not None:
            reasons.append(f"Parking fee: ₹{option['parking_fee']}")
        return ". ".join(reasons) + "."
