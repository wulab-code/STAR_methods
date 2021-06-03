function [filtered_image filtered_nuclei] = filter_network_output(segmented_image)

    
            dC = double(segmented_image); %this is the image array of categories (1-4)
        
            % now get the cell interior
            b_dC = dC == 2; % on dC, get all borders
            o_dC = dC >=2; % get all borders+ cell+nuclei                       
            i_dC = o_dC-b_dC; % subtract off borders to create cells
            
            % label each cell. 
            li_dC = bwlabel(i_dC); 

            regions = regionprops(li_dC,'Area');
            cell_areas = [regions(:).Area];
            
            % get cytoplasm
            n_dC = dC == 3;
            ln_dC = bwlabel(n_dC);
            regions_n = regionprops(ln_dC,'Area');
            m_area = mean([regions_n(:).Area]);
            
            % filter out all areas < expected minimum cell size (2x nucleus)
            li_dC = filtercellareas(cell_areas,2*m_area,li_dC);
            
            lio_dC = imfill(li_dC,'holes');
            
           % draw a boundary around each cell
            BW2 = bwperim(lio_dC,8);
                        
                        
            % reset all li_dC == 2;
            lio_dC  = lio_dC *2;
            
            % add back in cell boundaries
            bli_dC = BW2 + lio_dC;
            ind = bli_dC == 3;
            bli_dC(ind) = 1;
            
            % nuclei
            m_dC = dC == 4;
            lm_dC = bwlabel(m_dC);
            
            % filter out small nuclei
            region_m = regionprops(lm_dC,'Area');
            m_area = mean([region_m(:).Area]);
            m_dC = filtercellareas([region_m(:).Area],0.5*m_area,lm_dC);
            
            % 
            m_dC = m_dC .* lio_dC;
            
            m_dC = m_dC*4;
            
           
            % find all pixels that are in m_dC and replace them in li_dC;
            ind = m_dC ~=0;
            
            bli_dC(ind) = 3;
            filtered_image = bli_dC;
            filtered_nuclei = m_dC;
            
            
           
