import streamlit as st
from datetime import datetime, timedelta
import json
import re
import sys
import os
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from utils.api_helpers import get_weather_forecast, get_location_coordinates
from streamlit_folium import folium_static
import folium
from geopy.geocoders import Nominatim

class TravelChatbot:
    def __init__(self):
        try:
            if "GROQ_API_KEY" not in st.secrets:
                st.error("GROQ_API_KEY not found in Streamlit secrets")
                raise ValueError("GROQ_API_KEY not configured")
                
            self.llm = ChatGroq(
                temperature=0.2,
                api_key=st.secrets["GROQ_API_KEY"],
                model_name="llama-3.3-70b-versatile"
            )
        except Exception as e:
            st.error(f"Failed to initialize ChatGroq: {str(e)}")
            raise
        self.geolocator = Nominatim(user_agent="travel_chatbot")
        
    def get_location_coordinates(self, place):
        try:
            location = self.geolocator.geocode(place)
            if location:
                return {"lat": location.latitude, "lng": location.longitude}
        except:
            pass
        return None

    def display_map(self, destination, locations=[]):
        coords = self.get_location_coordinates(destination)
        if coords:
            m = folium.Map(location=[coords["lat"], coords["lng"]], zoom_start=12)
        else:
            m = folium.Map(location=[20, 0], zoom_start=2)

        for loc in locations:
            if "coordinates" in loc:
                try:
                    lat, lng = map(float, loc["coordinates"].split(","))
                    folium.Marker(
                        location=[lat, lng],
                        popup=loc["location"],
                        tooltip=loc["location"]
                    ).add_to(m)
                except (ValueError, TypeError):
                    st.warning(f"Invalid coordinates for location: {loc['location']}")

        folium_static(m)

    def extract_json_from_response(self, content):
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            if json_end > json_start:
                json_str = content[json_start:json_end].strip()
                return json_str
                
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            if json_end > json_start:
                json_str = content[json_start:json_end].strip()
                return json_str
                
        json_pattern = r'({[\s\S]*})'
        matches = re.search(json_pattern, content, re.DOTALL)
        if matches:
            return matches.group(1).strip()
            
        return content

    def generate_itinerary(self):
        if 'travel_info' not in st.session_state:
            st.session_state.travel_info = {}
        
        info = {
            'destination': st.session_state.chat_state['city'],
            'start_date': st.session_state.chat_state['start_date'],
            'end_date': st.session_state.chat_state['end_date'],
            'duration': st.session_state.chat_state['duration'],
            'arrival_time': st.session_state.chat_state['arrival_time'].strftime('%I:%M %p'),
            'departure_time': st.session_state.chat_state['departure_time'].strftime('%I:%M %p'),
            'companions': [st.session_state.chat_state['traveling_with']],
            'interests': st.session_state.chat_state['interests']
        }
        
        st.session_state.travel_info.update(info)
        
        try:
            city = info['destination']
            weather = get_weather_forecast(city)
        except Exception as e:
            weather = "Weather data not available"
            st.warning("Could not fetch weather data")
        
        try:
            system_message = f"""You are a smart AI travel assistant. 
            Generate a detailed itinerary for {info['destination']} from {info['start_date']} to {info['end_date']}, based on these interests: {', '.join(info['interests'])}.
            The trip is planned for {info['duration']} days.
            The user is traveling with: {', '.join(info['companions'])}.
            Arrival time on first day: {info['arrival_time']}
            Departure time on last day: {info['departure_time']}
            Consider the weather forecast: {weather}
            
            Important timing notes:
            - On the first day, only plan activities after the arrival time
            - On the last day, only plan activities before the departure time
            - For other days, plan a full day of activities
            
            YOU MUST RESPOND WITH ONLY A VALID JSON OBJECT, with no additional text before or after. 
            The response must follow this exact structure:
            {{
                "daily_plans": [
                    {{
                        "day": 1,
                        "activities": [
                            {{
                                "time": "14:00",
                                "activity": "description",
                                "location": "place name",
                                "coordinates": "lat,lng",
                                "weather": "weather forecast"
                            }}
                        ]
                    }}
                ]
            }}"""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Generate my travel itinerary in JSON format only."}
            ]
            
            response = self.llm.invoke(messages)
            
            if not response.content:
                st.error("Received empty response from the API.")
                return None
            
            json_str = self.extract_json_from_response(response.content)
            
            try:
                itinerary = json.loads(json_str)
                
                if "daily_plans" not in itinerary:
                    if "tripDetails" in itinerary or "days" in itinerary or "itinerary" in itinerary:
                        converted_itinerary = {"daily_plans": []}
                        
                        if "days" in itinerary:
                            for day in itinerary["days"]:
                                day_num = day.get("day", 1)
                                activities = day.get("activities", [])
                                converted_day = {"day": day_num, "activities": activities}
                                converted_itinerary["daily_plans"].append(converted_day)
                        elif "itinerary" in itinerary:
                            for i, day in enumerate(itinerary["itinerary"], 1):
                                converted_day = {"day": i, "activities": day.get("activities", [])}
                                converted_itinerary["daily_plans"].append(converted_day)
                        else:
                            st.warning("Received unexpected JSON structure. Attempting to use as-is.")
                            converted_itinerary = {"daily_plans": [itinerary]}
                        
                        itinerary = converted_itinerary
                st.session_state.travel_info['itinerary'] = itinerary
                st.session_state.travel_info['famous_places'] = self.fetch_famous_places(info['destination'])
                return itinerary
                
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse JSON response: {str(e)}")
                st.code(json_str)
                return None
            
        except Exception as e:
            st.error(f"Error generating itinerary: {str(e)}")
            return None
            
    def edit_itinerary(self, query):
        if 'itinerary' not in st.session_state.travel_info:
            return "I don't have an itinerary to edit yet. Please generate one first."
        
        try:
            current_itinerary = json.dumps(st.session_state.travel_info['itinerary'], indent=2)
            
            system_message = f"""You are a helpful travel assistant modifying an existing itinerary.
            The user has the following itinerary for {st.session_state.travel_info['destination']}:
            
            {current_itinerary}
            
            The user wants to modify this itinerary with this request: "{query}"
            
            YOU MUST RESPOND WITH ONLY A VALID JSON OBJECT containing the complete updated itinerary, with no additional text before or after.
            Your response must follow this exact structure:
            {{
                "daily_plans": [
                    {{
                        "day": 1,
                        "activities": [
                            {{
                                "time": "14:00",
                                "activity": "description",
                                "location": "place name",
                                "coordinates": "lat,lng",
                                "weather": "weather forecast"
                            }}
                        ]
                    }}
                ]
            }}"""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Update the itinerary with these changes: {query}"}
            ]
            
            response = self.llm.invoke(messages)
            
            json_str = self.extract_json_from_response(response.content)
            
            try:
                updated_itinerary = json.loads(json_str)
                
                if "daily_plans" not in updated_itinerary:
                    return f"I received an invalid response format. Please try again with more specific instructions. Please ensure your request is clear and specific."
                
                st.session_state.travel_info['itinerary'] = updated_itinerary
                
                return f"I've updated your itinerary based on your request: '{query}'. You can see the changes in the itinerary section."
                
            except json.JSONDecodeError:
                return f"I couldn't process the itinerary update. Here's what I understand about your request: {response.content}"
            
        except Exception as e:
            return f"I encountered an error while trying to update your itinerary: {str(e)}. Please try again later."
            
    def fetch_famous_places(self, city): 
        if "FOURSQUARE_API_KEY" not in st.secrets:
            st.error("FOURSQUARE_API_KEY not found in Streamlit secrets")
            raise ValueError("API keys not configured")

        api_key = st.secrets["FOURSQUARE_API_KEY"]
        url = f"https://api.foursquare.com/v2/venues/explore?near={city}&client_id={api_key}&v=20230101"
        
        response = requests.get(url)
        if response.status_code == 200:
            places = response.json().get("results", [])
            famous_places = []
            for place in places:
                name = place.get("name")
                address = place.get("formatted_address")
                image_url = place.get("imageUrl")
                famous_places.append({
                    "name": name,
                    "address": address,
                    "image_url": image_url
                })
            return famous_places
        else:
            st.error("Failed to fetch famous places.")
            return []

def suggest_locations(self, city, query=None):
    famous_places = self.fetch_famous_places(city)
    if famous_places and not query:
        places_info = "\n".join([f"{place['name']} - {place['address']} ![Image]({place['image_url']})" 
                            for place in famous_places if place['image_url']])
        return f"Here are some famous places in {city}:\n{places_info}"
    elif not query:
        return "No famous places found."
    
    try:
        info = st.session_state.travel_info
        
        system_message = f"""You are a helpful travel assistant suggesting additional locations for a trip to {info['destination']}.
        The user is interested in: {', '.join(info['interests'])}.
        The user request is: "{query}"
        
        YOU MUST RESPOND WITH ONLY A VALID JSON OBJECT containing suggested locations, with no additional text before or after.
        The response must follow this exact structure:
        {{
            "suggested_locations": [
                {{
                    "name": "Location name",
                    "description": "Brief description",
                    "reason": "Why it matches user interests",
                    "coordinates": "lat,lng",
                    "weather": "weather forecast"
                    
                }}
            ]
        }}"""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
        
        response = self.llm.invoke(messages)
        
        json_str = self.extract_json_from_response(response.content)
        
        try:
            suggestions = json.loads(json_str)
            
            if "suggested_locations" not in suggestions:
                return f"I received invalid suggestions. Please try again with more specific instructions."
            
            if 'suggested_locations' not in st.session_state.travel_info:
                st.session_state.travel_info['suggested_locations'] = []
            
            st.session_state.travel_info['suggested_locations'].extend(
                suggestions.get('suggested_locations', [])
            )
            
            location_list = []
            for loc in suggestions.get('suggested_locations', []):
                location_list.append(f"• **{loc['name']}**: {loc['description']}")
            
            return f"I've added these new locations to your map:\n\n" + "\n\n".join(location_list)
            
        except json.JSONDecodeError:
            return f"I couldn't process the location suggestions. Here's what I understand about your request: {response.content}"
            
    except Exception as e:
        return f"I encountered an error while suggesting locations: {str(e)}"


def show_chatbot():
    st.title("🌍 ChaloChalein")
    
    if 'travel_info' not in st.session_state:
        st.session_state.travel_info = {}
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if "chat_state" not in st.session_state:
        today = datetime.today().date()
        st.session_state.chat_state = {
            "step": "city",
            "city": "",
            "start_date": today,
            "end_date": today + timedelta(days=1),
            "duration": 1,
            "arrival_time": datetime.now().time(),
            "traveling_via": "vehicle",
            "departure_time": datetime.now().time(),
            "traveling_with": "No",
            "interests": [],
            "suggested_locations": []
        }

    main_col1, main_col2 = st.columns([2, 1])

    with main_col1:
        st.write("Plan your trip step by step!")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        chat_state = st.session_state.chat_state
        if chat_state["step"] == "city":
            city_input = st.text_input("🌍 Enter your destination city:", value=chat_state["city"])
            if st.button("Enter") and city_input:
                weather_data = get_weather_forecast(city_input)
                if weather_data:
                    if "error" in weather_data:
                        st.warning(weather_data["error"])
                    else:
                        st.write(f"Weather in {weather_data['location']}: {weather_data['temperature']}°C")
                        st.session_state.weather_data_displayed = True
                else:
                    st.warning("Could not retrieve weather data.")

                st.session_state.enter_button_pressed = True

            if st.session_state.get("weather_data_displayed", False) and st.button("Next"):
                chat_state.update({"city": city_input, "step": "dates"})

        elif chat_state["step"] == "dates":
            if st.button("🔙 Back", key="back_dates"):
                chat_state["step"] = "city"
                city_input = chat_state["city"]
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("📅 Start Date", value=chat_state["start_date"])
            with col2:
                end_date = st.date_input("📅 End Date", value=chat_state["end_date"], min_value=start_date)
            
            if start_date and end_date:
                if start_date > end_date:
                    st.error("Start date must be before the end date.")
                else:
                    duration = (end_date - start_date).days + 1
                    st.info(f"Trip duration: {duration} days")
                    if st.button("Confirm dates"):
                        chat_state.update({
                            "start_date": start_date,
                            "end_date": end_date,
                            "duration": duration,
                            "step": "traveling_via"
                        })

        elif chat_state["step"] == "traveling_via":
            if st.button("🔙 Back", key="back_traveling_via"):
                chat_state["step"] = "dates"
                traveling_via_input = chat_state["traveling_via"]
            traveling_via_input = st.radio("Do you already have planned your travel vehicle?", ["Yes", "No"])
            if traveling_via_input == "Yes":
                st.write("Great! Please let me know how you plan to travel.")
                traveling_via = st.text_input("Traveling via:")
            else:
                st.write("No worries! I can help you find a suitable travel vehicle.")
                st.write("Here are some ways to travel to your destination:")
                travel_options = ["Train", "Flight", "Bus"]
                selected_option = st.selectbox("Select a mode of transportation:", travel_options)
                traveling_via = selected_option
                if selected_option == "Train":
                    st.write("You can book train tickets on websites like [Trainline](https://www.thetrainline.com) or [Amtrak](https://www.amtrak.com).")
                elif selected_option == "Flight":
                    st.write("You can book flights on websites like [Skyscanner](https://www.skyscanner.com) or [Kayak](https://www.kayak.com).")
                elif selected_option == "Bus":
                    st.write("You can book bus tickets on websites like [Greyhound](https://www.greyhound.com) or [FlixBus](https://www.flixbus.com).")
            if st.button("Next"):
                chat_state.update({"traveling_via": traveling_via, "step": "times"})

        elif chat_state["step"] == "times":
            if st.button("🔙 Back", key="back_times"):
                chat_state["step"] = "traveling_via"
                arrival_time = chat_state["arrival_time"]
                departure_time = chat_state["departure_time"]
            st.write("### Travel Times")
            col1, col2 = st.columns(2)
            with col1:
                arrival_time = st.time_input(
                    "🛬 What time do you arrive?", 
                    value=chat_state["arrival_time"],
                    step=300
                )
                st.info(f"Arrival: {arrival_time.strftime('%I:%M %p')}")

            with col2:
                departure_time = st.time_input(
                    "🛫 What time do you depart?", 
                    value=chat_state["departure_time"],
                    step=300
                )
                st.info(f"Departure: {departure_time.strftime('%I:%M %p')}")
            if st.button("Confirm Times"):
                chat_state.update({
                    "arrival_time": arrival_time,
                    "departure_time": departure_time,
                    "step": "traveling_with"
                })

        elif chat_state["step"] == "traveling_with":
            if st.button("🔙 Back", key="back_traveling_with"):
                chat_state["step"] = "times"
                traveling_with = chat_state["traveling_with"]
            traveling_with = st.radio("Are you traveling with pets or children?", ["Yes", "No"])
            if st.button("Next"):
                chat_state.update({"traveling_with": traveling_with, "step": "interests"})

        elif chat_state["step"] == "interests":
            if st.button("🔙 Back", key="back_interests"):
                chat_state["step"] = "traveling_with"
                interests_input = ", ".join(chat_state["interests"])
            interests_input = st.text_area("🎯 Enter your interests (comma-separated)",
                                            value=", ".join(chat_state["interests"]))
            if st.button("Enter Interests"):
                if interests_input:
                    chat_state.update(
                        {"interests": [i.strip() for i in interests_input.split(",") if i.strip()], "step": "confirm"})

        elif chat_state["step"] == "confirm":
            if st.button("🔙 Back", key="back_interests"):
                chat_state["step"] = "interests"
            st.write("### Trip Summary")
            st.write(f"🌍 Destination: {chat_state['city']}")
            st.write(f"📅 Dates: {chat_state['start_date'].strftime('%B %d, %Y')} - {chat_state['end_date'].strftime('%B %d, %Y')}")
            st.write(f"⏱️ Duration: {chat_state['duration']} days")
            st.write(f"🚗 Traveling via: {chat_state['traveling_via']}")
            st.write(f"🛬 Arrival Time: {chat_state['arrival_time'].strftime('%I:%M %p')}")
            st.write(f"🛫 Departure Time: {chat_state['departure_time'].strftime('%I:%M %p')}")
            st.write(f"👥 Traveling with pets/children: {chat_state['traveling_with']}")
            st.write(f"🎯 Interests: {', '.join(chat_state['interests'])}")
            
            if st.button("Generate Itinerary"):
                chat_state["step"] = "generate"
                st.session_state.step = "itinerary"

        elif chat_state["step"] == "generate":
            if st.button("🔙 Back"):
                chat_state["step"] = "confirm"
                
            try:
                with st.spinner("Generating your personalized itinerary..."):
                    chatbot = TravelChatbot()
                    itinerary = chatbot.generate_itinerary()
                    
                if itinerary:
                    if 'suggested_locations' not in st.session_state.travel_info:
                        st.session_state.travel_info['suggested_locations'] = []

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Here's your personalized itinerary!"
                    })
                    chat_state["step"] = "itinerary"
                    st.rerun()
                else:
                    st.error("Failed to generate itinerary. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

        elif chat_state["step"] == "itinerary":
            if st.button("💾 Save Trip"):
                st.session_state.saved_trip = st.session_state.travel_info['itinerary']
                st.success("Trip saved successfully!")
            if 'itinerary' in st.session_state.travel_info and st.session_state.travel_info['itinerary']:
                itinerary = st.session_state.travel_info['itinerary']

            col1, col2 = st.columns(2)
            with col1:
                    if st.button("✏️ Edit Itinerary"):
                        st.session_state.edit_mode = True
            with col2:
                    if st.button("➕ More Suggestions"):
                        st.session_state.messages.append({
                            "role": "user", 
                            "content": "Can you suggest more locations to visit based on my interests?"
                        })
                        chatbot = TravelChatbot()
                        response = chatbot.suggest_locations("Suggest additional locations based on my interests")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        st.rerun()
                    
            for day in itinerary['daily_plans']:
                    with st.expander(f"Day {day['day']}", expanded=True):
                        for activity in day['activities']:
                            st.write(f"**{activity['time']}**: {activity['activity']}")
                            st.write(f"📍 Location: {activity['location']}")
                            
    with main_col2:
        if 'travel_info' in st.session_state and 'destination' in st.session_state.travel_info:
            st.header("📍 Recommended Locations")
            chatbot = TravelChatbot()
            
            locations = []
            
            if 'itinerary' in st.session_state.travel_info:
                for day in st.session_state.travel_info['itinerary']['daily_plans']:
                    locations.extend(day['activities'])
            
            if 'suggested_locations' in st.session_state.travel_info:
                for loc in st.session_state.travel_info['suggested_locations']:
                    locations.append({
                        "location": loc["name"],
                        "coordinates": loc["coordinates"]
                    })
            
            chatbot.display_map(st.session_state.travel_info['destination'], locations)
            chat_state["step"] = "itinerary"
            
    if chat_state["step"] == "itinerary":
        prompt = st.chat_input("Ask me anything about your itinerary or type 'edit' to modify it")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            chatbot = TravelChatbot()
            
            if "edit" in prompt.lower() or "change" in prompt.lower() or "modify" in prompt.lower():
                response = chatbot.edit_itinerary(prompt)
            elif "suggest" in prompt.lower() or "more location" in prompt.lower() or "additional place" in prompt.lower():
                response = chatbot.suggest_locations(prompt)
            else:
                response = chatbot.llm.invoke(prompt).content
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.session_state.travel_info['itinerary'] = itinerary
            st.rerun()


show_chatbot = show_chatbot
