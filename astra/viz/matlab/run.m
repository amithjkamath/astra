% function [outputArg1,outputArg2] = visualize(type_in, psize_in, nr_in, met_in, org_in, file)

close all
% handles = findall(groot, 'Type', 'figure', 'Tag', 'volshow');
% close(handles);
clear all

%%% For fast Data output into excel table


% type_in;  t     % 1,2 -> "Erosion", "Dilation"
% psize_in; s     % [1, 5]   -> 2, 3, 4, 5, 6
% nr_in;    n     % [1, 30]   -> "DLDP_070", "DLDP_071", "DLDP_072", 
%                                "DLDP_073", "DLDP_074", "DLDP_075", 
%                                "DLDP_076", "DLDP_078", "DLDP_079", 
%                                "DLDP_080", "DLDP_081", "DLDP_082", 
%                                "DLDP_083", "DLDP_084", "DLDP_085", 
%                                "DLDP_086", "DLDP_087", "DLDP_088",
%                                "DLDP_089", "DLDP_090", "DLDP_091"
% met_in;   m     % [1, 4]   -> "Max", "Mean", "DMax", "DMean"
% org_in;   o     % [1, 10]  -> tv, bs, hl, hr, el, er, chiasm, opnl, opnr, brain
% file      f     % 1,2 -> CTV, PTV


%% write into Excel file
% for t = 1:2
% %     for s = 1:4
%         for n = 1:4
%             for m = 1:4
%                 for o = 1:9
%                     writeEx(t, 5, n, m, o, 1);
%                 end
%             end
%         end
% %     end
% end

% 
% 
% for t = 1:2
%     for n = 19:20
%         for m = 1:4
%             for o = 1:9
%                 writeEx3(t, 21, m, o, 1);
%             end
%         end
%     end
% end

t = 1;
s = 2;
n = 11;
m = 4; %  problem with transparency effect visual in DMean = 4
o = 8; % problem with transparency effect visual in 8 or 9
f = 1;
% 
% 
%% show volshow
visualize(t, s, n, m, o, f );

% 
%% show 2D contour
% ContVisualize(t, s, n, m, o, f);


% sub_er_dil(s, n, m, o, f)


%% Information
%%% For PTV already evaluated: E3 D3
%%% For CTV already evaluated: E3 D3 E5 D5





