import whisper
import json
import tqdm
import openai
import os
import re
import csv
import numpy as np
import copy
from openai import OpenAI

def get_asr_result(audio_path, file_path, asr_processed_path, save_file=False):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            result = json.load(file)
    else:
        # use asr to get sentence from speech
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True)
        new_entry = {'audio_path': audio_path}
        result = {**new_entry, **result}

        if save_file:
            with open(file_path, "w") as file:
                json.dump(result, file, indent=4)

    parts = re.split('(\.{3}|[?!.])', result["text"])
    # Merge every two elements in the list (sentence and separator)
    merged_parts = [parts[i] + (parts[i+1] if i+1 < len(parts) else '') for i in range(0, len(parts), 2)]
    res_sentence = [part.strip() for part in merged_parts if part]

    fps = 60
    down_sample_rate = 8
    seg_data = result['segments']

    sentences_time_info = []
    sentence_info = []
    for seg_data_seq in seg_data:
        for word in seg_data_seq['words']:
            word_dict = {
                'word': word['word'].replace(" ", ""),
                'start_code': round(word['start']*fps/down_sample_rate),
                'end_code': round(word['end']*fps/down_sample_rate)
            }
            sentence_info.append(word_dict)
            if '.' in word['word'] or '?' in word['word'] or '!' in word['word']:
                sentence_dict = {
                    'sentence': sentence_info,
                    'sentence_start_code': sentence_info[0]['start_code'],
                    'sentence_end_code': sentence_info[-1]['end_code']
                }
                sentences_time_info.append(sentence_dict)
                sentence_info = []
    if sentence_info:
        sentence_dict = {
            'sentence': sentence_info,
            'sentence_start_code': sentence_info[0]['start_code'],
            'sentence_end_code': sentence_info[-1]['end_code']
        }
        sentences_time_info.append(sentence_dict)

    with open(asr_processed_path, "w") as file:
        json.dump(sentences_time_info, file, indent=4)

    if len(res_sentence) != len(sentences_time_info):
        print(audio_path)
        print(len(res_sentence), len(sentences_time_info))
        print(res_sentence)
    # assert(len(res_sentence) == len(sentences_time_info))

    return res_sentence, sentences_time_info


def get_llm_result(res_sentence, llm_file_path, model_path, save_file=False, type='paragraph'):
    if os.path.exists(llm_file_path):
        with open(llm_file_path, 'r', encoding='utf-8') as file:
            llm_label_res = json.load(file)
    else:
        api_key = "EMPTY"
        base_url = "http://localhost:8000/v1"
        client = OpenAI(
            api_key = api_key,
            base_url = base_url,
        )

        instruction = """You are a body language savvy, open, expressive speaker, adept at using your body language and gestures to impress your audience. You need to fully understand the semantics, emotion, timing, and coherence of a semantic gesture. You'll be given a speech and you'll label each place you should gesture (down to the word) with the corresponding gesture label, as shown here: 

        Hello (3123 HAND WAVE)! I wonder whether everybody (1211 FOREFINGER POINT) knows the meaning of these two words of "energy-conservation" and "low carbon"? 

        Imagine yourself gesturing while speaking, and position the semantic gestures as accurately as possible, as if a real person would be doing the semantic gesture here.

        The following are the gestures you can use, so please use only those that are already available, don't make them up.

        Semantic gestures:

        1120 ARM FLEX\t3120 FOREHEAD SALUTE\t3110 ARM RAISE HIGH-LEVEL\t3130 ARM RAISE MID-LEVEL\t1121 ARM WEIGHTLIFT\t2000 ARMS AKIMBO\t2030 ARMS FOLD\t2001 ARMS RAISE TOWORDS-SKY\t3100 ARMS RAISE V-SHAPE\t3111 ARMS REACH\t3330 HANDS SHELTER\t0020 ARMS SELF-EMBRACE\t1122 ARMS RUN\t3101 ARMS WELCOME\t1100 ARMS EXPLODE\t3010 HANDS RISE\t2220 ARMS DESCEND\t3030 ARMS SPHERICAL\t3310 ARMS FUSE\t2002 ARMS WING\t3311 ARMS SURROUND\t0030 BELLY PAT\t0031 BELLY RUB\t0032 BELLY PREGNANT\t0310 EYEBROW PRESS\t0300 CHEEK BRUSH\t0130 CHEEK SLAP\t0320 CHEEK SUPPORT\t1130 CHEST BEAT\t3000 CHEST HOLD\t0200 CHEST POINT\t0301 CHIN FLICK\t0302 CHIN POINT\t0330 CHIN RUB\t0201 EAR CUP\t0220 EARS BLOCK\t0210 EYE 'TELESCOPE'\t0311 EYE WIPE\t1210 FOREFINGER-AND-MIDDLE-FINGER GAZE\t0202 EYES RING\t0131 FACE COVER\t0221 FACE BARRIER\t3320 FINGERS BECKON\t1010 FINGERS SNAP\t2100 FINGERS SHUT\t2330 FINGERS TALK\t3121 FINGERS WAVE\t1000 FINGERS CROWD-COMPACT\t3210 FINGERTIPS KISS\t1330 FINGERTIPS RUB\t1131 FIST BEAT\t1132 FIST CLENCH\t1133 FIST PUNCH\t2221 ELBOW FALL\t1020 FISTS COMBAT\t1110 FISTS WRING\t1111 FISTS COLLISON\t0211 FOOT TAP\t1011 FOREFINGER BEAT\t1230 FOREFINGER HOP\t1211 FOREFINGER POINT\t3131 FOREFINGER RAISE\t1231 FOREFINGER RAISE-ONE\t3102 FOREFINGER RAISE-SKY\t2101 FOREFINGER WAG\t2310 FOREFINGER SPIN\t1232 FOREFINGER SPIRAL\t1030 FOREFINGER-AND-MIDDLE-FINGER POINT\t0000 FOREFINGER-AND-MIDDLE-FINGER'SMOKE'\t1012 FOREFINGER-AND-MIDDLE-FINGER STAB\t3122 FOREFINGER-AND-MIDDLE-FINGER SALUTE\t1031 FOREFINGER-AND-MIDDLE-FINGER SCISSORS\t1220 FOREFINGER-AND-MIDDLE-FINGER STEPS\t2120 FOREFINGERS AIM\t3312 FOREFINGERS HOOK\t1212 FOREFINGERS POINT-FORWARD\t1310 WRISTS CHANGE\t1213 FOREFINGERS MEASURE\t0110 FOREHEAD SLAP\t0111 FOREHEAD PRESS\t0112 FOREHEAD FINGER-TAP\t0132 FOREHEAD HAND-TAP\t0113 FOREHEAD WIPE\t2210 HAND CHOP\t3321 HAND CALL\t3220 HAND 'DRINK'\t0021 HAND FAN\t2110 HAND FLAP\t3331 HAND FLOP\t1013 HAND JAB\t2320 HAND MEASURE-DOWN\t3011 HAND MEASURE-UP\t0010 HAND PURSE-AROUND\t0011 HAND PURSE\t3230 HAND RING\t2311 HAND ROTATE\t1001 HAND SNATCH\t2331 HAND TOSS\t3231 HAND V-SIGN\t2130 HAND WAG\t3123 HAND WAVE\t3300 HAND 'WRITE'\t2300 PALM HALT\t0212 WRIST CHECK-TIME\t2111 FOREARM THROW-SIDE\t2131 FOREARM REPULSE\t2211 FOREARM CUT\t1101 FIST KNOCK\t0001 TEETH BRUSH\t0230 PALM BARRIER\t3332 HAND SAFEGUARD\t2020 HAND OPEN\t1331 PALM SAW\t3020 HAND DOORWAY-TURN\t1311 HAND HIP-HOP\t2132 HANDS CROSS\t1320 HANDS 'FLUTE'\t2200 HANDS SCISSOR\t2021 HANDS SHRUG\t2230 HANDS 'THROTTLE'\t2301 HANDS T-SIGN\t1332 HANDS APPLAUSE\t3021 HANDS EMPHASIS\t1221 FOREFINGER SCAN\t1222 PALMS TURN-PAGE\t2212 PALMS CROSSCUT\t1312 FINGERS KEYBOARD\t2112 PALMS REPEL\t3301 HANDS EXPLAIN\t1300 HANDS SHOOT\t1321 HANDS PERCUSSION\t1322 HANDS STRUM\t1323 HANDS SERENADE\t3211 HANDS DRAW-BACK\t2222 PALMS OVERTURN\t2332 FOREFINGER EMPTY\t3031 PALMS EXPAND\t1301 HANDS DRAW-OUTLINE\t3200 FINGERS 'LOVE'\t1302 HANDS STEERING\t2201 HANDS DIVIDE\t1021 FIST CLASP\t1022 HANDS BEAST\t0002 HANDS GAME-HANDLE\t3022 HANDS REVEAL\t3221 HANDS FEED\t1032 FINGERS AIR-QUOTES\t2010 HEAD NOD\t0331 HEAD ROLL\t0100 HEAD SCRATCH\t2022 HEAD SHAKE\t0303 HAIR FROOM\t0203 EARS BUNNY\t3201 HEART CLASP\t3202 HEART CROSS\t0231 MOUTH SILENCE\t0232 LIPS ZIP\t0120 MOUTH CLASP\t0022 MOUTH FAN\t0321 MOUTH SHIELD\t2231 NECK CLAMP\t0023 NECK RUB\t0332 NECK SCRATCH\t0312 NOSE FAN\t0222 NOSE TOUCH\t2321 PALM LOWER\t1112 PALM PUNCH\t2113 PALM THRUST\t3001 PALM UP\t1102 PALMS BRUSH\t3132 PALMS CONTACT\t2031 PALMS FRONT\t3112 PALMS UP-HIGH\t2312 PALMS UP-LOW\t0012 PALMS RUB\t2313 PALMS ABSTENTION\t2032 PALMS BOUNDARY\t0322 SHOULDERS SHRUG\t0101 TEMPLE CIRCLE\t0121 TEMPLE 'SHOOT'\t0102 TEMPLE TOUCH\t2202 THROAT 'CUT'\t0122 THROAT GRASP\t2121 THUMB DOWN\t3322 THUMB HITCH\t2122 THUMB POINT\t3232 THUMB UP\t2322 FINGERS ESTIMATE\t1200 THUMB, FOREFINGER AND LITTLE-FINGER RAISE\t1002 WAIST OUTLINE\t1201 FINGERS 'THREE'\t1202 FINGERS 'FIVE'\t2023 PALMS REVERSE\t3002 PALM OFFER\t3313 HANDS UNITE\t2232 HEAD SURRENDER\t2302 ARM ENDEAVOR\t3032 ARMS ENCOMPASS\tPALM RISE\t3222 HAND TOAST

        Now you will be given a speech sentence, and you'll label each place you should gesture with the corresponding gesture label mentioned above. please only use the gestures that are already available, and output the give speech sentence with labels, don't output other words.

        """
        llm_label_res = []

        if type == 'sentence':
            for sample in res_sentence:
                completion = client.chat.completions.create(
                    model = model_path,
                    messages = [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": sample}
                    ]
                )
                llm_label_res.append(completion.choices[0].message.content)
        elif type == 'paragraph':
            paragraph_str = ''
            for sample in res_sentence:
                paragraph_str += sample
            completion = client.chat.completions.create(
                model = model_path,
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": paragraph_str}
                ]
            )
            annotated_paragraph_str = completion.choices[0].message.content
            annotated_paragraph_str = annotated_paragraph_str.replace('\n', '')
            parts = re.split('(?<=[?!.])', annotated_paragraph_str)
            llm_label_res = [part.strip() for part in parts if part]
        else:
            raise ValueError('type must be sentence or paragraph')
    
        if save_file:
            with open(llm_file_path, "w") as file:
                json.dump(llm_label_res, file, indent=4)

    return llm_label_res

def get_gesture_vqcode_reflection(gesture_vqcode_path):
    categories = {}
    with np.load(gesture_vqcode_path, allow_pickle=True) as file:
        arrays_dict = {name: file[name] for name in file.files}
        for filename, value in arrays_dict.items():
            category_number = str(int((filename.split('-')[0]).split('_')[1]))
            if category_number not in categories:
                categories[category_number] = []

            gesture_code_dict = {
                'file_name': filename.split('_')[1],
                'code_info': value,
                'length': value[0].shape[-1]
            }
            categories[category_number].append(gesture_code_dict)

    return categories

def find_swap_vq_code_info(audio_path, gesture_vqcode_path, model_path, save_dir, delete_a_pose=False):
    asr_file_path = os.path.join(save_dir, 'asr_results.json')
    asr_processed_path = os.path.join(save_dir, 'asr_processed_results.json')
    llm_file_path = os.path.join(save_dir, 'llm_results.json')
    
    res_sentence, sentences_time_info = get_asr_result(audio_path, asr_file_path, asr_processed_path, 1)
    llm_label_res = get_llm_result(res_sentence, llm_file_path, model_path, 1, 'sentence')
    
    if not delete_a_pose:
        gesture_vqcode_reflection = get_gesture_vqcode_reflection(gesture_vqcode_path)
    else:
        gesture_vqcode_reflection = delete_a_pose_code(gesture_vqcode_path)

    insert_sg_info = []
    for i in range(len(llm_label_res)):
        modified_s_0 = re.sub(r'\s([,.!?])', r'\1', llm_label_res[i])
        modified_s_1 = re.sub(r'(?<!\s)\(', ' (', modified_s_0)
        print(modified_s_1)

        pattern = re.compile(r'\((.*?)\)')
        matches = pattern.finditer(modified_s_1)

        words_num_in_brackets = 0
        for match in matches:
            content = match.group(1)
            start_position = match.start()
            substring_before_bracket = modified_s_1[:start_position]
            words_before_bracket = substring_before_bracket.split()
            last_word = words_before_bracket[-1]
            word_index = len(words_before_bracket) - 1 - words_num_in_brackets 

            sg_index = content.split()[0]
            if sg_index not in gesture_vqcode_reflection:
                print('sg_index {} not in gesture_vqcode_reflection'.format(sg_index))
                words_num_in_brackets += len(content.split())
                continue
                
            sg_info = gesture_vqcode_reflection[sg_index]
            print(i, word_index, last_word)
            print(len(sentences_time_info[i]['sentence']))
            insert_motion_dict = {
                'semantic_gesture_index': sg_index,
                'semantic_gesture_label': content.split()[1:],
                'semantic_gesture_info': sg_info,
                'sentence_index': i,
                'word_index': word_index,
                'last_word': last_word,
                'start_code': sentences_time_info[i]['sentence'][word_index]['start_code'],
                'end_code': sentences_time_info[i]['sentence'][word_index]['end_code'],
                'sentence_start_code': sentences_time_info[i]['sentence_start_code'],
                'sentence_end_code': sentences_time_info[i]['sentence_end_code']
            }
            insert_sg_info.append(insert_motion_dict)
            words_num_in_brackets += len(content.split())

    return insert_sg_info


def delete_a_pose_code(gesture_vqcode_path):
    gesture_vqcode_reflection = get_gesture_vqcode_reflection(gesture_vqcode_path)
    # print(gesture_vqcode_reflection)
    all_length = 0
    all_start_length = 0
    all_end_length = 0

    up_code_count = np.zeros((1024), dtype = int)
    start_end_code_sum = np.zeros((1024), dtype = int)
    start_end_code = []
    for key, value in gesture_vqcode_reflection.items():
        for sg in value:
            upper_code = sg['code_info'][0]
            first_upper_code = upper_code[0]
            last_upper_code = upper_code[-1]
            for code in upper_code:
                up_code_count[code] += 1
            start_end_code_sum[first_upper_code] += 1
            start_end_code_sum[last_upper_code] += 1

        for i in range(len(start_end_code_sum)):
            if start_end_code_sum[i] > 10:
                start_end_code.append(i)

    top_30 = np.argsort(up_code_count)[-30:][::-1]
    a_pose_code = set(top_30).intersection(set(start_end_code))

    delete_dive_reserve = 0
    reserve_length = []
    str_file = ''
    for key, value in gesture_vqcode_reflection.items():
        for sg in value:
            upper_code = sg['code_info'][0]
            first_upper_code = upper_code[0]
            last_upper_code = upper_code[-1]
            start_repeat_num = 1
            last_repaet_num = 1
            i = 1
            while upper_code[i] == first_upper_code and i < len(upper_code)-1:
                start_repeat_num += 1
                i += 1
            j = -2
            while upper_code[j] == last_upper_code and j >= -len(upper_code)+1:
                last_repaet_num += 1
                j -= 1

            if sg['file_name'] == '167-0' or sg['file_name'] == '165-0':
                sg['code_info'] = sg['code_info'][:, start_repeat_num-1:sg['length']-last_repaet_num+1]
                sg['length'] = sg['length'] - start_repeat_num - last_repaet_num + 2
                assert(sg['length'] == sg['code_info'][0].shape[0])
                reserve_length.append(sg['length'])
                continue

            while upper_code[j] in a_pose_code and j >= -len(upper_code)+1:
                start_repeat_num += 1
                j -= 1
            while upper_code[i] in a_pose_code and i < len(upper_code)-1:
                last_repaet_num += 1
                i += 1

            if start_repeat_num > 1 and last_repaet_num > 1:
                sg['code_info'] = sg['code_info'][:, start_repeat_num-2:sg['length']-last_repaet_num+2]
                sg['length'] = sg['length'] - start_repeat_num - last_repaet_num + 4
            else:
                sg['code_info'] = sg['code_info'][:, start_repeat_num-1:sg['length']-last_repaet_num+1]
                sg['length'] = sg['length'] - start_repeat_num - last_repaet_num + 2

            # print(sg['length'], sg['code_info'][0].shape[0])
            assert(sg['length'] == sg['code_info'][0].shape[0])
            reserve_length.append(sg['length'])

            all_length += len(upper_code)
            all_start_length += start_repeat_num
            all_end_length += last_repaet_num

            now_dive_value = (start_repeat_num+last_repaet_num) / len(upper_code)
            if now_dive_value > delete_dive_reserve:
                delete_dive_reserve = now_dive_value
                str_file = sg['file_name']
    # print("delete_dive_reserve", delete_dive_reserve, str_file)
    print((all_start_length+all_end_length)/all_length)

    # print(len(a_pose_code))
    print(np.mean(reserve_length)*8/60, np.std(reserve_length))

    return gesture_vqcode_reflection

def insert_sg_2part(motion_vq_info, insert_sg_info, save_dir, delete_a_pose=False, sg_type='vq'):
    if delete_a_pose:
        insert_info_file_path = os.path.join(save_dir, 'insert_info_without_a_pose.json')
    else:
        insert_info_file_path = os.path.join(save_dir, 'insert_info_with_a_pose.json')
    save_insert_info = []

    body_vq, hands_vq = motion_vq_info
    new_body_vq = copy.deepcopy(body_vq)
    new_hands_vq = copy.deepcopy(hands_vq)

    last_insert_pos = 0
    for sg_info in insert_sg_info:
        sg_motion_num = len(sg_info['semantic_gesture_info'])
        choice_index = np.random.randint(0, sg_motion_num)
        choice = sg_info['semantic_gesture_info'][choice_index]['code_info'] 
        insert_motion_len = sg_info['semantic_gesture_info'][choice_index]['length']
        start_code = sg_info['start_code']
        end_code = sg_info['end_code']
        sentence_start_code = sg_info['sentence_start_code']
        sentence_end_code = sg_info['sentence_end_code']

        # insert in the end
        start_change_pos = max(sentence_start_code, end_code-round(insert_motion_len*0.75), last_insert_pos+4)
        end_change_pos = start_change_pos + insert_motion_len

        print(start_change_pos, end_change_pos, insert_motion_len, sg_info['last_word'])

        if sg_type == 'vq':
            reserve_vq_len = len(new_body_vq) - start_change_pos
            if insert_motion_len > reserve_vq_len:
                end_change_pos = len(new_body_vq)
                insert_motion_len = reserve_vq_len
                new_body_vq[start_change_pos:end_change_pos] = choice[0][0:reserve_vq_len]
                new_hands_vq[start_change_pos:end_change_pos] = choice[1][0:reserve_vq_len]
                insert_sg_s_e_code = choice[:, [0, reserve_vq_len-1]]
            else:
                new_body_vq[start_change_pos:end_change_pos] = choice[0]
                new_hands_vq[start_change_pos:end_change_pos] = choice[1]
                insert_sg_s_e_code = choice[:, [0, -1]]
        elif sg_type == 'rq':
            # new_body_vq: [4, len]
            # choice: [3, 4, 1, insert_motion_len]
            reserve_vq_len = new_body_vq.shape[1] - start_change_pos
            if reserve_vq_len <= 0:
                continue
            if insert_motion_len > reserve_vq_len:
                end_change_pos = new_body_vq.shape[1]
                insert_motion_len = reserve_vq_len
                new_body_vq[:, start_change_pos:end_change_pos] = choice[0][:, 0, 0:reserve_vq_len]
                new_hands_vq[:, start_change_pos:end_change_pos] = choice[1][:, 0, 0:reserve_vq_len]
                insert_sg_s_e_code = choice[:, :, 0, [0, reserve_vq_len-1]]
            else:
                print(choice[0].shape, new_body_vq[:, start_change_pos:end_change_pos].shape)
                new_body_vq[:, start_change_pos:end_change_pos] = choice[0][:, 0]
                new_hands_vq[:, start_change_pos:end_change_pos] = choice[1][:, 0]
                insert_sg_s_e_code = choice[:, :, 0, [0, -1]]
        else:
            raise ValueError('sg_type must be vq or rq')
        
        # print(insert_sg_s_e_code)
        save_insert_info.append({
            'insert_last_word': sg_info['last_word'],
            'start_change_pos': start_change_pos, 
            'end_change_pos': end_change_pos, 
            'sg_index': sg_info['semantic_gesture_index'], 
            'sg_label': sg_info['semantic_gesture_label'],
            'choice_index': choice_index,
            'sg_length': insert_motion_len,
            'insert_sg_s_e_code': insert_sg_s_e_code.tolist()
        })

        last_insert_pos = end_change_pos

    with open(insert_info_file_path, "w") as file:
        json.dump(save_insert_info, file, indent=4)

    return new_body_vq, new_hands_vq, save_insert_info
