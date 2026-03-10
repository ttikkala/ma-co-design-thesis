# This script was written by generative AI (Claude model Sonnet 4.5, accessed: Sep 8, 2025)

# Modify leg lengths of the ant robot's xml file to create the desired morphology that can be fed to the simulator

import xml.etree.ElementTree as ET
import numpy as np
import os
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SegmentLengthParams:
    """Parameters for a single leg segment"""
    min_length: float = 0.2828
    max_length: float = 0.2828
    distribution: str = 'normal'
    mean_length: Optional[float] = None  # If None, uses midpoint of range
    std_length: Optional[float] = None   # If None, uses range/6

    def __post_init__(self):
        """Set default values if not provided"""
        if self.mean_length is None:
            self.mean_length = (self.min_length + self.max_length) / 2
        if self.std_length is None:
            self.std_length = (self.max_length - self.min_length) / 6

@dataclass
class AntLegParams:
    """Parameters for ant leg modification with segment-level control"""
    # Front legs: aux segment, leg segment, ankle segment
    front_aux: SegmentLengthParams
    front_leg: SegmentLengthParams  
    front_ankle: SegmentLengthParams
    
    # Back legs: aux segment, leg segment, ankle segment
    back_aux: SegmentLengthParams
    back_leg: SegmentLengthParams
    back_ankle: SegmentLengthParams
    
class AntLegModifier:
    """Class to modify ant leg lengths in MuJoCo XML files with segment-level control"""
    
    def __init__(self, xml_file: str):
        """Initialize with the path to the ant XML file"""
        self.xml_file = xml_file
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()
        
        # Define leg segments by position and segment type
        # Structure: {position: {segment_type: [leg_names]}}
        self.leg_structure = {
            'front': {
                'aux': ['front_left_aux_geom', 'front_right_aux_geom'],
                'leg': ['front_left_leg_geom', 'front_right_leg_geom'],
                'ankle': ['front_left_ankle_geom', 'front_right_ankle_geom']
            },
            'back': {
                'aux': ['back_left_aux_geom', 'back_right_aux_geom'],
                'leg': ['back_left_leg_geom', 'back_right_leg_geom'],
                'ankle': ['back_left_ankle_geom', 'back_right_ankle_geom']
            }
        }
        
        # Map individual legs to their segments
        self.leg_segments = {
            'front_left': ['front_left_aux_geom', 'front_left_leg_geom', 'front_left_ankle_geom'],
            'front_right': ['front_right_aux_geom', 'front_right_leg_geom', 'front_right_ankle_geom'],
            'back_left': ['back_left_aux_geom', 'back_left_leg_geom', 'back_left_ankle_geom'],
            'back_right': ['back_right_aux_geom', 'back_right_leg_geom', 'back_right_ankle_geom']
        }
        
        # Store original values
        self.original_values = self._extract_original_values()
        self.original_segment_lengths = self._calculate_original_segment_lengths()
    
    def _extract_original_values(self) -> Dict:
        """Extract original fromto values and positions from the XML"""
        values = {}
        
        # Extract geom fromto values
        for geom in self.root.iter('geom'):
            name = geom.get('name')
            if name and any(name in segments for segments in self.leg_segments.values()):
                fromto = geom.get('fromto')
                if fromto:
                    values[name] = fromto
        
        # Extract body positions
        for body in self.root.iter('body'):
            name = body.get('name')
            if name and ('leg' in name or 'foot' in name or 'aux' in name):
                pos = body.get('pos')
                if pos:
                    values[f"{name}_pos"] = pos
        
        # Extract site positions for touch sensors
        for site in self.root.iter('site'):
            name = site.get('name')
            if name and 'touch' in name:
                pos = site.get('pos')
                if pos:
                    values[f"{name}_pos"] = pos
        
        return values
    
    def _calculate_original_segment_lengths(self) -> Dict[str, float]:
        """Calculate the length of each segment in the original model"""
        segment_lengths = {}
        
        # Calculate length for each individual segment
        for leg_name, segments in self.leg_segments.items():
            for segment_name in segments:
                for geom in self.root.iter('geom'):
                    if geom.get('name') == segment_name:
                        fromto = geom.get('fromto')
                        if fromto:
                            coords = self._parse_coordinates(fromto)
                            if len(coords) == 6:
                                start = np.array(coords[:3])
                                end = np.array(coords[3:])
                                segment_length = np.linalg.norm(end - start)
                                segment_lengths[segment_name] = segment_length
                        break
        
        return segment_lengths
    
    def _parse_coordinates(self, coord_str: str) -> List[float]:
        """Parse coordinate string into list of floats"""
        return [float(x) for x in coord_str.split()]
    
    def _format_coordinates(self, coords: List[float]) -> str:
        """Format list of floats back to coordinate string"""
        return ' '.join(f"{x:.6f}" for x in coords)
    
    def _scale_vector(self, vector: List[float], scale_factor: float) -> List[float]:
        """Scale a 3D vector by a given factor"""
        return [v * scale_factor for v in vector]
    
    def _generate_length_from_params(self, params: SegmentLengthParams) -> float:
        """Generate a single length value from parameters"""
        if params.distribution == 'normal':
            length = np.random.normal(params.mean_length, params.std_length)
        elif params.distribution == 'uniform':
            length = np.random.uniform(params.min_length, params.max_length)
        elif params.distribution == 'lognormal':
            sigma = params.std_length / params.mean_length
            mu = np.log(params.mean_length) - 0.5 * sigma**2
            length = np.random.lognormal(mu, sigma)
        else:
            raise ValueError(f"Unsupported distribution: {params.distribution}")
        
        # Clamp to limits
        return np.clip(length, params.min_length, params.max_length)
    
    def modify_leg_lengths(self, params: AntLegParams) -> Dict[str, Dict]:
        """
        Modify leg lengths according to segment-specific parameters.
        Maintains left-right symmetry.
        
        Args:
            params: AntLegParams object with segment specifications
            
        Returns:
            Dictionary with detailed modification information
        """
        modifications = {
            'segments': {},
            'legs': {}
        }
        
        # Generate target lengths for each segment type (maintaining symmetry)
        segment_target_lengths = {
            'front_aux': self._generate_length_from_params(params.front_aux),
            'front_leg': self._generate_length_from_params(params.front_leg),
            'front_ankle': self._generate_length_from_params(params.front_ankle),
            'back_aux': self._generate_length_from_params(params.back_aux),
            'back_leg': self._generate_length_from_params(params.back_leg),
            'back_ankle': self._generate_length_from_params(params.back_ankle)
        }
        
        # Calculate scale factors for each segment type
        segment_scales = {}
        
        # Front segments (left and right use same scale)
        for segment_type in ['aux', 'leg', 'ankle']:
            segment_key = f'front_{segment_type}'
            target_length = segment_target_lengths[segment_key]
            
            # Get original length from left leg (both sides are same in original)
            geom_name = self.leg_structure['front'][segment_type][0]  # left
            original_length = self.original_segment_lengths[geom_name]
            
            scale = target_length / original_length if original_length > 0 else 1.0
            segment_scales[segment_key] = scale
            
            modifications['segments'][segment_key] = {
                'target_length': target_length,
                'original_length': original_length,
                'scale_factor': scale
            }
            
            # Apply to both left and right
            for geom_name in self.leg_structure['front'][segment_type]:
                self._modify_geom_segment(geom_name, scale)
        
        # Back segments (left and right use same scale)
        for segment_type in ['aux', 'leg', 'ankle']:
            segment_key = f'back_{segment_type}'
            target_length = segment_target_lengths[segment_key]
            
            # Get original length from left leg
            geom_name = self.leg_structure['back'][segment_type][0]  # left
            original_length = self.original_segment_lengths[geom_name]
            
            scale = target_length / original_length if original_length > 0 else 1.0
            segment_scales[segment_key] = scale
            
            modifications['segments'][segment_key] = {
                'target_length': target_length,
                'original_length': original_length,
                'scale_factor': scale
            }
            
            # Apply to both left and right
            for geom_name in self.leg_structure['back'][segment_type]:
                self._modify_geom_segment(geom_name, scale)
        
        # Update body positions and touch sites for each leg
        # Front left
        self._modify_body_positions_with_segment_scales(
            'front_left',
            segment_scales['front_aux'],
            segment_scales['front_leg'],
            segment_scales['front_ankle']
        )
        
        # Front right
        self._modify_body_positions_with_segment_scales(
            'front_right',
            segment_scales['front_aux'],
            segment_scales['front_leg'],
            segment_scales['front_ankle']
        )
        
        # Back left
        self._modify_body_positions_with_segment_scales(
            'back_left',
            segment_scales['back_aux'],
            segment_scales['back_leg'],
            segment_scales['back_ankle']
        )
        
        # Back right
        self._modify_body_positions_with_segment_scales(
            'back_right',
            segment_scales['back_aux'],
            segment_scales['back_leg'],
            segment_scales['back_ankle']
        )
        
        # Calculate total leg lengths for reporting
        for leg_name in self.leg_segments.keys():
            position = 'front' if 'front' in leg_name else 'back'
            original_total = sum(
                self.original_segment_lengths[seg] 
                for seg in self.leg_segments[leg_name]
            )
            target_total = sum(
                segment_target_lengths[f'{position}_{seg_type}']
                for seg_type in ['aux', 'leg', 'ankle']
            )
            
            modifications['legs'][leg_name] = {
                'original_total_length': original_total,
                'target_total_length': target_total
            }
        
        return modifications
    
    def _modify_geom_segment(self, geom_name: str, scale: float):
        """Modify a specific geom segment's fromto attribute"""
        for geom in self.root.iter('geom'):
            if geom.get('name') == geom_name:
                fromto = geom.get('fromto')
                if fromto:
                    coords = self._parse_coordinates(fromto)
                    if len(coords) == 6:
                        start = coords[:3]
                        end = coords[3:]
                        direction = [end[i] - start[i] for i in range(3)]
                        scaled_direction = self._scale_vector(direction, scale)
                        new_end = [start[i] + scaled_direction[i] for i in range(3)]
                        new_coords = start + new_end
                        geom.set('fromto', self._format_coordinates(new_coords))
                break
    
    def _modify_body_positions_with_segment_scales(self, leg_name: str, 
                                                   aux_scale: float,
                                                   leg_scale: float, 
                                                   ankle_scale: float):
        """Modify body positions using segment-specific scales"""
        # Map leg names to body name patterns
        body_info = {
            'front_left': [
                ('front_left_aux', aux_scale),
                ('front_left_foot', aux_scale * leg_scale)
            ],
            'front_right': [
                ('front_right_aux', aux_scale),
                ('front_right_foot', aux_scale * leg_scale)
            ],
            'back_left': [
                ('back_left_aux', aux_scale),
                ('back_left_foot', aux_scale * leg_scale)
            ],
            'back_right': [
                ('back_right_aux', aux_scale),
                ('back_right_foot', aux_scale * leg_scale)
            ]
        }
        
        if leg_name in body_info:
            for body_name, scale in body_info[leg_name]:
                for body in self.root.iter('body'):
                    if body.get('name') == body_name:
                        pos = body.get('pos')
                        if pos:
                            coords = self._parse_coordinates(pos)
                            scaled_coords = self._scale_vector(coords, scale)
                            body.set('pos', self._format_coordinates(scaled_coords))
                        break
            
            # Modify touch sites
            self._modify_touch_sites_with_segment_scales(
                leg_name, aux_scale, leg_scale, ankle_scale
            )
    
    def _modify_touch_sites_with_segment_scales(self, leg_name: str,
                                                aux_scale: float,
                                                leg_scale: float,
                                                ankle_scale: float):
        """Modify touch site positions using segment-specific scales"""
        site_info = {
            'front_left': [
                ('front_left_leg_touch', aux_scale),
                ('front_left_ankle_touch', aux_scale * leg_scale)
            ],
            'front_right': [
                ('front_right_leg_touch', aux_scale),
                ('front_right_ankle_touch', aux_scale * leg_scale)
            ],
            'back_left': [
                ('back_left_leg_touch', aux_scale),
                ('back_left_ankle_touch', aux_scale * leg_scale)
            ],
            'back_right': [
                ('back_right_leg_touch', aux_scale),
                ('back_right_ankle_touch', aux_scale * leg_scale)
            ]
        }
        
        if leg_name in site_info:
            for site_name, scale in site_info[leg_name]:
                for site in self.root.iter('site'):
                    if site.get('name') == site_name:
                        pos = site.get('pos')
                        if pos:
                            coords = self._parse_coordinates(pos)
                            scaled_coords = self._scale_vector(coords, scale)
                            site.set('pos', self._format_coordinates(scaled_coords))
                        break
    
    def save_modified_xml(self, output_file: str):
        """Save the modified XML to a new file"""
        ET.indent(self.tree, space="  ", level=0)
        self.tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
    def reset_to_original(self):
        """Reset the XML back to original values"""
        self.tree = ET.parse(self.xml_file)
        self.root = self.tree.getroot()
        self.original_segment_lengths = self._calculate_original_segment_lengths()


def generate_ants(params_list, log_timestamp, agent_id, input_file="/home/tiia/thesis/bodies/ant_highres.xml", output_dir="/home/tiia/thesis/bodies/xmls"):
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Show original segment lengths
    modifier = AntLegModifier(input_file)
    print("Original segment lengths:")
    for position in ['front', 'back']:
        for segment_type in ['aux', 'leg', 'ankle']:
            geom_name = modifier.leg_structure[position][segment_type][0]
            length = modifier.original_segment_lengths[geom_name]
            print(f"  {position}_{segment_type}: {length:.4f}\n")

    params = AntLegParams(front_aux=SegmentLengthParams(), 
                          front_leg=SegmentLengthParams(min_length=params_list[0], max_length=params_list[0]),
                          front_ankle=SegmentLengthParams(min_length=params_list[1], max_length=params_list[1]),
                          back_aux=SegmentLengthParams(),
                          back_leg=SegmentLengthParams(min_length=params_list[2], max_length=params_list[2]),
                          back_ankle=SegmentLengthParams(min_length=params_list[3], max_length=params_list[3]))
    
    modifications = modifier.modify_leg_lengths(params)
    
    output_file = os.path.join(output_dir, f"ant_{log_timestamp}_{agent_id}.xml")
    modifier.save_modified_xml(output_file)
    
    print(f"Generated {output_file}")
    print("  Segment modifications:")
    for seg_name, info in modifications['segments'].items():
        print(f"    {seg_name}: {info['original_length']:.4f} -> "
                f"{info['target_length']:.4f} (scale: {info['scale_factor']:.3f})")
    print("  Total leg lengths:")
    for leg_name, info in modifications['legs'].items():
        print(f"    {leg_name}: {info['original_total_length']:.4f} -> "
                f"{info['target_total_length']:.4f}")

    return output_file


