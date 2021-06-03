 function imdata = cell_decomposition(bw_filtered_image, bw_filtered_nuclei,orig,cfp,expand)  

            
            % segment cells - save bounding box, pixeldxlist, and pixel intensities
            % From each segmented image, get all the pixels/intensities from the
            % original CFP and Venus (labeled FRET) images. Save the pixel locations.
            % Use this (instead of bounding box) to link up segmented cells
            for l = 1:max(bw_filtered_image(:))
                % use bounding box to get region of cell
%                 bb = enforceboundariesrect(round(regions2(l).BoundingBox),size(matfile.var));    
                % get foreground and background of jth cell - expand minicells
                
                cur_cell_pixels = find(bw_filtered_image == l);
                cur_nuc_pixels = find(bw_filtered_nuclei == l);
                if isempty(cur_nuc_pixels)
                    cur_nuc_pixels = cur_cell_pixels;
                end
                [venusimout, cfpimout, venusimbg, cfpimbg, bbout, pixelidx, newarea, nucleusim] = ...
                    getminicell(orig,cfp,cur_cell_pixels,expand,cur_nuc_pixels);
                                
                imdata(l).venuscell = venusimout;
                imdata(l).cfpcell = cfpimout;
                imdata(l).venusbg = venusimbg;
                imdata(l).cfpbg = cfpimbg;
                imdata(l).Area = newarea;
                imdata(l).PixelIdxList = pixelidx;
                imdata(l).BoundingBox = bbout;
                imdata(l).nucleus = nucleusim;
                imdata(l).NucleiPixels = cur_nuc_pixels;                                
                
            end                        
            
        