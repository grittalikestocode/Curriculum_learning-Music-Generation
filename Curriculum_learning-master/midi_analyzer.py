import os
import pretty_midi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde, entropy
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from math import log2
from scipy.integrate import quad


class MidiAnalyzer:
    """
    Comprehensive analyzer for MIDI files that calculates various musical metrics
    including rhythmic intensity and pitch class entropy.
    """
    
    def __init__(self, default_beats_per_bar=4, default_subbeats_per_beat=4):
        """
        Initialize the analyzer with default time signature parameters
        
        Args:
            default_beats_per_bar: Default number of beats in a bar (typically 4 for 4/4 time)
            default_subbeats_per_beat: Default number of sub-divisions per beat (e.g., 4 for 16th notes)
        """
        self.default_beats_per_bar = default_beats_per_bar
        self.default_subbeats_per_beat = default_subbeats_per_beat

    def extract_pitch_sequence(self,midi_data):
        notes = []
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
            notes.extend(instrument.notes)
        notes.sort(key=lambda n: n.start)
        return [note.pitch for note in notes]

    def extract_pitch_class_sequence(self,midi_data, resolution=0.25):
        """
        Quantize notes into pitch classes at fixed time intervals.
        Returns a list of pitch class sets per time bin.
        """
        end_time = midi_data.get_end_time()
        time_bins = np.arange(0, end_time + resolution, resolution)
        sequence = []
    
        for t_start in time_bins:
            t_end = t_start + resolution
            pitches = set()
            for instrument in midi_data.instruments:
                if instrument.is_drum:
                    continue
                for note in instrument.notes:
                    if note.start < t_end and note.end > t_start:
                        pitches.add(note.pitch % 12)
            sequence.append(pitches)
    
        return sequence
    
    
    def load_midi(self, midi_path):
        """
        Load a MIDI file
        
        Args:
            midi_path: Path to the MIDI file
            
        Returns:
            PrettyMIDI object
        """
        try:
            return pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            print(f"Error loading MIDI file {midi_path}: {e}")
            return None
    
    def calculate_rhythmic_intensity(self, midi_data, beats_per_bar=None, subbeats_per_beat=None):
        """
        Calculate rhythmic intensity as the percentage of sub-beats with at least one note onset.
        
        The formula is:
        RhythmicIntensity = (1/B) * Σ(1 if onset_count_b ≥ 1 else 0) for b in range(B)
        
        where B is the number of sub-beats in a bar and the indicator function
        returns 1 if there's at least one onset in the sub-beat.
        
        Args:
            midi_data: PrettyMIDI object
            beats_per_bar: Number of beats in a bar (defaults to class default)
            subbeats_per_beat: Number of sub-divisions per beat (defaults to class default)
            
        Returns:
            Average rhythmic intensity score (0-1)
        """
        if beats_per_bar is None:
            beats_per_bar = self.default_beats_per_bar
        if subbeats_per_beat is None:
            subbeats_per_beat = self.default_subbeats_per_beat
        
        # Get beat times
        try:
            beats = midi_data.get_beats()
            if len(beats) < 2:
                return 0  # Not enough beats to analyze
        except:
            return 0  # Could not extract beats
        
        # Calculate sub-beat times
        subbeat_times = []
        for i in range(len(beats) - 1):
            beat_duration = beats[i+1] - beats[i]
            subbeat_duration = beat_duration / subbeats_per_beat
            for j in range(subbeats_per_beat):
                subbeat_times.append(beats[i] + j * subbeat_duration)
        
        # If last beat doesn't have explicit next beat, extrapolate subbeats
        if len(beats) > 1:
            last_beat_duration = beats[-1] - beats[-2]
            for j in range(subbeats_per_beat):
                subbeat_times.append(beats[-1] + j * last_beat_duration / subbeats_per_beat)
        
        # Group sub-beats into bars
        subbeats_per_bar = beats_per_bar * subbeats_per_beat
        num_bars = max(1, len(subbeat_times) // subbeats_per_bar)
        
        # Count onsets for each sub-beat
        bar_intensities = []
        
        for bar in range(num_bars):
            bar_start_idx = bar * subbeats_per_bar
            bar_end_idx = min((bar + 1) * subbeats_per_bar, len(subbeat_times) - 1)
            if bar_end_idx <= bar_start_idx:
                continue
                
            # Initialize array to track if each sub-beat has at least one onset
            has_onset = np.zeros(bar_end_idx - bar_start_idx)
            
            # Check all instruments for note onsets
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    # Find which sub-beat this note onset falls into
                    onset_time = note.start
                    
                    # Skip notes outside the current bar
                    if onset_time < subbeat_times[bar_start_idx] or onset_time >= subbeat_times[bar_end_idx]:
                        continue
                        
                    # Find the sub-beat index for this onset
                    for sb_idx in range(bar_end_idx - bar_start_idx):
                        current_subbeat = bar_start_idx + sb_idx
                        next_subbeat = current_subbeat + 1
                        
                        # Handle edge case for last subbeat
                        if next_subbeat >= len(subbeat_times):
                            if onset_time >= subbeat_times[current_subbeat]:
                                has_onset[sb_idx] = 1
                            break
                            
                        # Normal case: check if onset falls within this subbeat
                        if subbeat_times[current_subbeat] <= onset_time < subbeat_times[next_subbeat]:
                            has_onset[sb_idx] = 1
                            break
            
            # Calculate intensity for this bar (percentage of subbeats with onsets)
            if len(has_onset) > 0:
                bar_intensity = np.sum(has_onset) / len(has_onset)
                bar_intensities.append(bar_intensity)
        
        # Return average rhythmic intensity across all bars
        return np.mean(bar_intensities) if bar_intensities else 0
    
    def calculate_pitch_class_entropy(self, midi_data, weighted_by_duration=True):
        """
        Calculate the entropy of the pitch class histogram.
        
        The formula is:
        PCH_Entropy = -Σ(p_i * log2(p_i)) for i in range(12)
        
        where p_i is the probability/frequency of pitch class i.
        Higher entropy indicates more uniform distribution of pitch classes.
        
        Args:
            midi_data: PrettyMIDI object
            weighted_by_duration: If True, weight pitch classes by note duration
            
        Returns:
            Pitch class histogram entropy (higher values indicate more diverse pitch usage)
        """
        # Initialize pitch class histogram (12 pitch classes from C to B)
        pitch_histogram = np.zeros(12)
        
        # Fill in the histogram
        total_weight = 0
        
        for instrument in midi_data.instruments:
            # Skip drum tracks
            if instrument.is_drum:
                continue
                
            for note in instrument.notes:
                # Get weight (duration or just count)
                weight = (note.end - note.start) if weighted_by_duration else 1
                
                # Convert MIDI pitch to pitch class (0-11)
                pitch_class = note.pitch % 12
                
                # Add to histogram
                pitch_histogram[pitch_class] += weight
                total_weight += weight
        
        # Normalize histogram to get probability distribution
        if total_weight > 0:
            pitch_histogram = pitch_histogram / total_weight
        
        # Calculate entropy (use base 2 for bits)
        # Only include non-zero entries to avoid log(0)
        nonzero_probs = pitch_histogram[pitch_histogram > 0]
        
        if len(nonzero_probs) > 0:
            # Direct calculation using formula: -Σ(p_i * log2(p_i))
            entropy_value = -np.sum(nonzero_probs * np.log2(nonzero_probs))
            
            # Normalize by maximum possible entropy (log2(12)) to get value between 0-1
            max_entropy = log2(12)  # Maximum entropy is when all 12 pitch classes are equally likely
            normalized_entropy = entropy_value / max_entropy
            
            return normalized_entropy
        else:
            return 0
    
    def calculate_note_density(self, midi_data):
        """
        Calculate the average number of notes per second
        
        Args:
            midi_data: PrettyMIDI object
            
        Returns:
            Notes per second
        """
        # Get total number of notes
        note_count = sum(len(instrument.notes) for instrument in midi_data.instruments)
        
        # Get duration in seconds
        duration = midi_data.get_end_time()
        
        # Calculate density (avoid division by zero)
        return note_count / max(duration, 0.001)
    
    def calculate_pitch_range(self, midi_data):
        """
        Calculate the span between highest and lowest notes
        
        Args:
            midi_data: PrettyMIDI object
            
        Returns:
            Pitch range in semitones
        """
        all_pitches = []
        
        for instrument in midi_data.instruments:
            if not instrument.is_drum:  # Skip drum tracks
                all_pitches.extend([note.pitch for note in instrument.notes])
        
        if not all_pitches:
            return 0
            
        return max(all_pitches) - min(all_pitches)

    def calculate_avg_pitch_interval(self, midi_data, track_idx=None):
        """
        Calculate average absolute pitch interval between consecutive notes in semitones.
        
        Args:
            midi_data: PrettyMIDI object
            track_idx: Specific track index (None for all non-drum tracks)
            
        Returns:
            Average pitch interval (float), or 0 if <2 notes found
        """
        try:
            intervals = []
            
            # Process specific track or all melodic tracks
            tracks = [midi_data.instruments[track_idx]] if track_idx is not None \
                   else [inst for inst in midi_data.instruments if not inst.is_drum]
            
            for instrument in tracks:
                if len(instrument.notes) < 2:
                    continue
                    
                # Sort notes by onset time
                sorted_notes = sorted(instrument.notes, key=lambda x: x.start)
                
                # Calculate intervals between consecutive pitches
                for i in range(1, len(sorted_notes)):
                    intervals.append(abs(sorted_notes[i].pitch - sorted_notes[i-1].pitch))
            
            return np.mean(intervals) if intervals else 0
            
        except Exception as e:
            print(f"Error calculating pitch intervals: {e}")
            return 0

    def calculate_avg_ioi(self, midi_data, track_idx=None):
        """
        Calculate average inter-onset interval (time between consecutive notes in seconds).
        
        Args:
            midi_data: PrettyMIDI object
            track_idx: Specific track index (None for all non-drum tracks)
            
        Returns:
            Average IOI (float), or 0 if <2 notes found
        """
        try:
            iois = []
            
            tracks = [midi_data.instruments[track_idx]] if track_idx is not None \
                   else [inst for inst in midi_data.instruments if not inst.is_drum]
            
            for instrument in tracks:
                if len(instrument.notes) < 2:
                    continue
                    
                # Get sorted onset times
                onsets = sorted([note.start for note in instrument.notes])
                iois.extend(np.diff(onsets))
            
            return np.mean(iois) if iois else 0
            
        except Exception as e:
            print(f"Error calculating IOIs: {e}")
            return 0

    def calculate_upc_per_bar(self, midi_data, min_notes=1):
        """
        Calculate Used Pitch Classes (UPC) per bar (0-12)
        
        Args:
            midi_data: PrettyMIDI object
            min_notes: Minimum notes required to count a bar
            
        Returns:
            List of UPC counts per bar, or empty list if no valid bars
        """
        try:
            beats = midi_data.get_beats()
            if len(beats) < 2:
                return []
                
            upc_counts = []
            beats_per_bar = self.default_beats_per_bar
            
            for i in range(0, len(beats)-1, beats_per_bar):
                bar_start = beats[i]
                bar_end = beats[min(i+beats_per_bar, len(beats)-1)]
                
                pitch_classes = set()
                for instr in midi_data.instruments:
                    if instr.is_drum:
                        continue
                        
                    for note in instr.notes:
                        if bar_start <= note.start < bar_end:
                            pitch_classes.add(note.pitch % 12)
                
                if len(pitch_classes) >= min_notes:
                    upc_counts.append(len(pitch_classes))
                    
            return upc_counts
            
        except Exception as e:
            print(f"UPC calculation error: {e}")
            return []

    def calculate_polyphony(self, midi_data, resolution=0.1):
        """
        Calculate average polyphony (simultaneous notes)
        
        Args:
            midi_data: PrettyMIDI object
            resolution: Time step in seconds for evaluation
            
        Returns:
            Average polyphony (float)
        """
        try:
            total_time = midi_data.get_end_time()
            if total_time <= 0:
                return 0.0
                
            time_steps = np.arange(0, total_time, resolution)
            polyphony = np.zeros(len(time_steps))
            
            for i, t in enumerate(time_steps):
                active_notes = 0
                for instr in midi_data.instruments:
                    if instr.is_drum:
                        continue
                    for note in instr.notes:
                        if note.start <= t < note.end:
                            active_notes += 1
                polyphony[i] = active_notes
                
            # Only consider time steps with at least 1 note
            meaningful_steps = polyphony[polyphony > 0]
            return np.mean(meaningful_steps) if len(meaningful_steps) > 0 else 0.0
            
        except Exception as e:
            print(f"Polyphony calculation error: {e}")
            return 0.0

    def calculate_polyphonic_rate(self, midi_data, resolution=0.1):
        total_steps = 0
        polyphonic_steps = 0
        total_time = midi_data.get_end_time()
        time_steps = np.arange(0, total_time, resolution)
    
        for t in time_steps:
            active_notes = sum(
                1 for instr in midi_data.instruments if not instr.is_drum
                for note in instr.notes if note.start <= t < note.end
            )
            if active_notes > 0:
                total_steps += 1
                if active_notes > 1:
                    polyphonic_steps += 1
    
        return polyphonic_steps / total_steps if total_steps else 0
         

    def calculate_chord_irregularity(self, midi_data, min_notes=3):
        """
        Calculate chord progression irregularity (% unique chord trigrams)
        
        Args:
            midi_data: PrettyMIDI object
            min_notes: Minimum notes to consider a chord
            
        Returns:
            Irregularity percentage (0-100)
        """
        try:
            # Extract chords (simplified - for production use a proper chord analyzer)
            chords = []
            for instr in midi_data.instruments:
                if instr.is_drum:
                    continue
                    
                # Group simultaneous notes into chords
                for t in [note.start for note in instr.notes]:
                    chord = set()
                    for note in instr.notes:
                        if abs(note.start - t) < 0.05:  # 50ms tolerance
                            chord.add(note.pitch % 12)
                    if len(chord) >= min_notes:
                        chords.append(frozenset(chord))
            
            # Count trigrams
            if len(chords) < 3:
                return 0.0
                
            trigrams = set()
            for i in range(len(chords)-2):
                trigram = (chords[i], chords[i+1], chords[i+2])
                trigrams.add(trigram)
            
            return 100 * len(trigrams) / (len(chords)-2)
            
        except Exception as e:
            print(f"Chord irregularity error: {e}")
            return 0.0

    def calculate_empty_bar_ratio(self, midi_data):
        tempo = midi_data.estimate_tempo()
        seconds_per_beat = 60.0 / tempo
        bar_length = seconds_per_beat * self.default_beats_per_bar
        total_bars = int(np.ceil(midi_data.get_end_time() / bar_length))
        bar_has_note = np.zeros(total_bars, dtype=bool)
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                bar_index = int(note.start / bar_length)
                if bar_index < total_bars:
                    bar_has_note[bar_index] = True
        empty_bars = np.sum(~bar_has_note)
        return 100 * empty_bars / total_bars if total_bars > 0 else 0
    
    def calculate_unique_pitches_durations(self, midi_data):
        pitches = set()
        durations = set()
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                pitches.add(note.pitch)
                durations.add(round(note.end - note.start, 5))
        return len(pitches), len(durations)
    
    # def calculate_qualified_note_ratio(self, midi_data, timestep=0.125):
    #     count = 0
    #     total = 0
    #     for instrument in midi_data.instruments:
    #         if instrument.is_drum:
    #             continue
    #         for note in instrument.notes:
    #             total += 1
    #             if (note.end - note.start) >= 3 * timestep:
    #                 count += 1
    #     return 100 * count / total if total else 0

    def calculate_qualified_note_ratio(self, midi_data):
        tempo = midi_data.estimate_tempo()
        beat = 60.0 / tempo
        timestep = beat / 8  # 32nd note
        threshold = 3 * timestep  # qualified if duration >= 3 timesteps
    
        count = 0
        total = 0
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                total += 1
                if (note.end - note.start) >= threshold:
                    count += 1
        return 100 * count / total if total else 0
    
    def calculate_tone_span(self, midi_data):
        pitches = [note.pitch for inst in midi_data.instruments if not inst.is_drum for note in inst.notes]
        return max(pitches) - min(pitches) if pitches else 0
    
    def calculate_pitch_repetition_features(self, midi_data, min_dur=1.0, jump_thresh=12):
        cpr = 0
        dpr = 0
        ts_jumps = 0
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
            sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
            prev_pitch = None
            duration_sum = 0
            for i, note in enumerate(sorted_notes):
                if note.pitch == prev_pitch:
                    cpr += 1
                    duration_sum += note.end - note.start
                else:
                    if duration_sum >= min_dur:
                        dpr += 1
                    duration_sum = note.end - note.start
                    if i > 0 and abs(note.pitch - sorted_notes[i-1].pitch) > jump_thresh:
                        ts_jumps += 1
                    prev_pitch = note.pitch
            if duration_sum >= min_dur:
                dpr += 1
        return cpr, dpr, ts_jumps
    
    # def calculate_qualified_rhythm_frequency(self, midi_data):
    #     valid = {1, 0.5, 0.25, 0.125, 0.0625}  # and dotted/triplet versions
    #     dotted = {x * 1.5 for x in valid}
    #     triplet = {x * 2/3 for x in valid}
    #     all_valid = valid | dotted | triplet
    #     count = 0
    #     total = 0
    #     for instrument in midi_data.instruments:
    #         if instrument.is_drum:
    #             continue
    #         for note in instrument.notes:
    #             dur = round(note.end - note.start, 4)
    #             total += 1
    #             if dur in all_valid:
    #                 count += 1
    #     return 100 * count / total if total else 0

    def calculate_qualified_rhythm_frequency(self, midi_data):
        tempo = midi_data.estimate_tempo()
        beat = 60.0 / tempo  # duration of a quarter note
    
        valid = {beat * x for x in [1, 0.5, 0.25, 0.125, 0.0625]}
        dotted = {x * 1.5 for x in valid}
        triplet = {x * 2/3 for x in valid}
        all_valid = valid | dotted | triplet
    
        count = 0
        total = 0
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                dur = round(note.end - note.start, 4)
                total += 1
                if any(abs(dur - v) < 0.01 for v in all_valid):
                    count += 1
        return 100 * count / total if total else 0
    
    def calculate_pitch_consonance_score(self, midi_data):
        consonant = {0, 3, 4, 7, 8, 9}
        dissonant = {1, 2, 6, 10, 11}
        total_score = 0
        total = 0
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
            sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
            for i in range(1, len(sorted_notes)):
                interval = abs(sorted_notes[i].pitch - sorted_notes[i-1].pitch) % 12
                if interval in consonant:
                    score = 1
                elif interval in dissonant:
                    score = -1
                else:
                    score = 0
                total_score += score
                total += 1
        return total_score / total if total else 0

    def calculate_IR(self,sequence):
        unique = sorted(list(set(sequence)))
        n = len(unique)
    
        # Initialize counts
        note_counts = defaultdict(int)
        transition_counts = {u: defaultdict(int) for u in unique}
        information_rates = []
    
        for i in range(1, len(sequence)):
            current = sequence[i]
            previous = sequence[i - 1]
    
            # Estimate probability distributions
            total_notes = i
            p_notes = np.array([note_counts[u] for u in unique], dtype=float)
            p_notes = (p_notes + 1e-6) / (total_notes + 1e-6 * n)  # add smoothing
            H_notes = entropy(p_notes, base=2)
    
            trans_total = sum(transition_counts[previous].values())
            p_trans = np.array([transition_counts[previous][u] for u in unique], dtype=float)
            p_trans = (p_trans + 1e-6) / (trans_total + 1e-6 * n)
            H_trans = entropy(p_trans, base=2)
    
            IR = max(H_notes - H_trans, 0)
            information_rates.append(IR)
    
            # Update counts
            note_counts[current] += 1
            transition_counts[previous][current] += 1
    
        return np.mean(information_rates) if information_rates else 0

    def calculate_information_rate(self, midi_data):
        sequence = self.extract_pitch_sequence(midi_data)
        return self.calculate_IR(sequence)

    def compute_ssm(self,sequence):
        """
        Compute self-similarity matrix from a sequence of pitch class sets.
        Converts to binary vectors and computes cosine similarity.
        """
        if not sequence:
            return np.array([])
    
        vecs = np.zeros((len(sequence), 12))
        for i, pc_set in enumerate(sequence):
            for pc in pc_set:
                vecs[i, pc] = 1
    
        return cosine_similarity(vecs)

    def calculate_ssm(self, midi_data, resolution=0.25):
        sequence = self.extract_pitch_class_sequence(midi_data, resolution)
        return self.compute_ssm(sequence)

    def calculate_ssm_score(self, midi_data):
        ssm = self.calculate_ssm(midi_data)
        if ssm.size == 0:
            return 0
        # Mean off-diagonal similarity (avoid perfect self-similarity)
        mask = ~np.eye(len(ssm), dtype=bool)
        return np.mean(ssm[mask])

    def plot_ssm(self,ssm,filepath, title="Self-Similarity Matrix"):
        plt.figure(figsize=(6, 6))
        plt.imshow(ssm, origin='lower', cmap='hot', interpolation='nearest')
        plt.title(title)
        plt.xlabel("Time bins")
        plt.ylabel("Time bins")
        plt.colorbar(label="Similarity")
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
    
    def analyze_midi_file(self, midi_path):
        """
        Perform comprehensive analysis on a single MIDI file
        
        Args:
            midi_path: Path to the MIDI file
            
        Returns:
            Dictionary of analysis results
        """
        midi_data = self.load_midi(midi_path)
        if midi_data is None:
            return None
            
        file_size = os.path.getsize(midi_path)
        duration = midi_data.get_end_time()
        note_count = sum(len(instrument.notes) for instrument in midi_data.instruments)

        # Calculate UPC first since it's used in multiple metrics
        upc_per_bar = self.calculate_upc_per_bar(midi_data)  # This line was missing
        
        # Calculate all metrics
        results = {
            'filename': os.path.basename(midi_path),
            'rhythmic_intensity': self.calculate_rhythmic_intensity(midi_data),
            'pitch_entropy': self.calculate_pitch_class_entropy(midi_data),
            'note_density': self.calculate_note_density(midi_data),
            'pitch_range': self.calculate_pitch_range(midi_data),
            'avg_pitch_interval': self.calculate_avg_pitch_interval(midi_data),
            'avg_ioi': self.calculate_avg_ioi(midi_data),
            'upc_mean': np.mean(upc_per_bar) if upc_per_bar else 0,
            'upc_std': np.std(upc_per_bar) if upc_per_bar else 0,
            'polyphony': self.calculate_polyphony(midi_data),
            'polyphonic_rate': self.calculate_polyphonic_rate(midi_data),
            'chord_irregularity': self.calculate_chord_irregularity(midi_data),
            'empty_bar_ratio': self.calculate_empty_bar_ratio(midi_data),
            'unique_pitch_duration':self.calculate_unique_pitches_durations(midi_data),
            'qualified_note_ratio':self.calculate_qualified_note_ratio(midi_data),
            'tone_span':self.calculate_tone_span(midi_data),
            'pitch_rep':self.calculate_pitch_repetition_features(midi_data),
            'qualified_rythm_freq':self.calculate_qualified_rhythm_frequency(midi_data),
            'pitch_consonance_score':self.calculate_pitch_consonance_score(midi_data),
            'information_rate': self.calculate_information_rate(midi_data),
            'ssm': self.calculate_ssm_score(midi_data),
            'duration': duration,
            'note_count': note_count,
            'file_size': file_size
        }
        

        return results
    
    def analyze_midi_collection(self, directory, output_csv=None, model_name=None):
        """
        Analyze all MIDI files in a directory
        
        Args:
            directory: Directory containing MIDI files
            output_csv: Optional path to save results as CSV
            model_name: Optional name of the model that generated these files (for grouping)
            
        Returns:
            DataFrame with analysis results
        """
        results = []
        
        for filename in os.listdir(directory):
            if filename.endswith('.mid') or filename.endswith('.midi'):
                filepath = os.path.join(directory, filename)
                try:
                    result = self.analyze_midi_file(filepath)
                    # pm = self.load_midi(filepath)
                    # ssm = self.calculate_ssm(pm)
                    # out_path = os.path.join(directory, filename.replace(".mid", "_ssm.png"))
                    # self.plot_ssm(ssm, filepath=out_path, title=filename)
                    # ssm_files.append(out_path)
                    if result:
                        if model_name:
                            result['model'] = model_name
                        results.append(result)
                        
                except Exception as e:
                    print(f"Error analyzing {filename}: {e}")

        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV if requested
        if output_csv and not df.empty:
            df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")
            
        return df
    
    def compare_models(self, model_directories, output_dir=None):
        """
        Compare metrics across different models
        
        Args:
            model_directories: Dictionary mapping model names to directories
            output_dir: Directory to save output files
            
        Returns:
            DataFrame with combined results
        """
        all_results = []
        
        for model_name, directory in model_directories.items():
            print(f"Analyzing model: {model_name}")
            df = self.analyze_midi_collection(
                directory, 
                output_csv=f"{output_dir}/{model_name}_analysis.csv" if output_dir else None,
                model_name=model_name
            )
            all_results.append(df)
        
        # Combine results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            combined_df.to_csv(f"{output_dir}/combined_analysis.csv", index=False)
            
            # Create comparison visualizations
            self.create_comparison_visualizations(combined_df, output_dir)
        
        return combined_df
    
    def create_comparison_visualizations(self, df, output_dir):
        """
        Create visualizations comparing metrics across models
        
        Args:
            df: DataFrame with analysis results including a 'model' column
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # # 1. Box plots for key metrics
        # metrics = ['rhythmic_intensity', 'pitch_entropy', 'note_density', 'duration']
        # plt.figure(figsize=(15, 10))
        
        # for i, metric in enumerate(metrics):
        #     plt.subplot(2, 2, i+1)
        #     df.boxplot(column=metric, by='model')
        #     plt.title(f'Distribution of {metric.replace("_", " ").title()}')
        #     plt.suptitle('')  # Remove default title
        #     plt.xticks(rotation=45)
        #     plt.tight_layout()
        
        # plt.savefig(f"{output_dir}/metrics_boxplot_comparison.png")
        # plt.close()
        
        # # 2. Scatter plot of rhythmic intensity vs pitch entropy
        # plt.figure(figsize=(10, 8))
        # models = df['model'].unique()
        # colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        # for i, model in enumerate(models):
        #     model_data = df[df['model'] == model]
        #     plt.scatter(
        #         model_data['rhythmic_intensity'], 
        #         model_data['pitch_entropy'],
        #         c=[colors[i]],
        #         label=model,
        #         alpha=0.7
        #     )
        
        # plt.xlabel('Rhythmic Intensity')
        # plt.ylabel('Pitch Class Entropy')
        # plt.title('Relationship between Rhythmic Intensity and Pitch Entropy')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.savefig(f"{output_dir}/rhythmic_vs_pitch_scatter.png")
        # plt.close()
        
        # # 3. Histograms for individual metrics
        # for metric in metrics:
        #     plt.figure(figsize=(12, 8))
        #     for i, model in enumerate(models):
        #         model_data = df[df['model'] == model]
        #         plt.hist(
        #             model_data[metric], 
        #             alpha=0.5, 
        #             label=model, 
        #             bins=20
        #         )
            
        #     plt.xlabel(metric.replace('_', ' ').title())
        #     plt.ylabel('Count')
        #     plt.title(f'Distribution of {metric.replace("_", " ").title()} Across Models')
        #     plt.legend()
        #     plt.savefig(f"{output_dir}/{metric}_histogram.png")
        #     plt.close()
        
        # 4. Summary statistics table
        # summary_df = df.groupby('model').agg({
        #     'rhythmic_intensity': ['mean', 'std', 'min', 'max'],
        #     'pitch_entropy': ['mean', 'std', 'min', 'max'],
        #     'note_density': ['mean', 'std', 'min', 'max'],
        #     'duration': ['mean', 'std', 'min', 'max'],
        #     'note_count': ['mean', 'sum']
        # })
        
        # summary_df.to_csv(f"{output_dir}/summary_statistics.csv")
        # Replace fixed metric list with dynamic numeric feature aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('file_size', errors='ignore')
        
        summary_df = df.groupby('model')[numeric_cols].agg(['mean', 'std', 'min', 'max'])
        
        summary_df.to_csv(f"{output_dir}/summary_statistics.csv")
        print(f"Saved full summary statistics to {output_dir}/summary_statistics.csv")


def main():
    parser = argparse.ArgumentParser(description='Analyze MIDI files for musical metrics')
    parser.add_argument('--file', help='Path to a single MIDI file to analyze')
    parser.add_argument('--dir', help='Directory of MIDI files to analyze')
    parser.add_argument('--output', help='Output CSV file path')
    parser.add_argument('--compare', nargs='+', help='Compare multiple directories (format: name:path name2:path2)')
    parser.add_argument('--output-dir', help='Directory to save all output files')
    parser.add_argument('--beats-per-bar', type=int, default=4, help='Number of beats per bar')
    parser.add_argument('--subbeats-per-beat', type=int, default=4, help='Number of subdivisions per beat')
    
    args = parser.parse_args()
    
    analyzer = MidiAnalyzer(
        default_beats_per_bar=args.beats_per_bar,
        default_subbeats_per_beat=args.subbeats_per_beat
    )
    
    if args.file:
        result = analyzer.analyze_midi_file(args.file)
        if result:
            print("\nAnalysis Results:")
            for key, value in result.items():
                print(f"{key}: {value}")
    
    elif args.dir:
        df = analyzer.analyze_midi_collection(args.dir, args.output)
        if not df.empty:
            print("\nSummary Statistics:")
            print(df.describe())
    
    elif args.compare:
        model_dirs = {}
        for model_dir in args.compare:
            parts = model_dir.split(':')
            if len(parts) == 2:
                model_dirs[parts[0]] = parts[1]
        
        if model_dirs:
            df = analyzer.compare_models(model_dirs, args.output_dir)
            print("\nComparison completed. Check output directory for results.")
    
    else:
        parser.print_help()

    


if __name__ == "__main__":
    main()
