function srep_minigrid_3d_twin
% SREP MINI-GRID — 3D DIGITAL TWIN (pure MATLAB, no toolboxes)
% =========================================================================
% Runs your REAL trained networks with plain matrix math:
%   * LSTM forecaster  -> hand-coded gate equations on the dumped weights
%   * residual-PPO     -> hand-coded MLP (tanh) on the dumped weights
% Pipeline each step:  webread weather (sensors) -> 24x7 window -> LSTM
%   forecast -> 60-d observation -> VecNormalize -> PPO action -> residual
%   over the load-following baseline -> battery -> 3D scene + HUD + alarm.
%
% Needs only base MATLAB (R2022b+ recommended for 3D in uifigure).
% Put srep_matlab_pack.mat (from dump_weights_for_matlab.py) in:
%   - the same folder as this file, OR ./matlab_export/  (auto-found)
% Then just run:  srep_minigrid_3d_twin
% -------------------------------------------------------------------------

P = loadPack();
sig = @(x) 1./(1+exp(-x));

% --- shared sim state ---
S.tierIdx = 2;                       % 1=50 2=75 3=120
S = applyTier(S);
S.hour = 6; S.soc = 0.55; S.soh = 1.0;
S.playing = true; S.speed = 1.5; S.muted = false;
S.day = synthDay(S);                 % until live weather loads
S.dataMsg = 'modeled day'; S.lastBeep = 0;
S.LOAD_SHAPE = [.45 .40 .38 .38 .42 .55 .85 1.05 .95 .82 .78 .80 ...
                .82 .80 .78 .82 .95 1.35 1.95 2.20 2.05 1.55 .95 .60];

ui = struct();
buildUI();
fetchWeather(6.30, 0.05, 'Akosombo');   % auto-load default town
runLoop();

% =====================================================================
% MAIN LOOP
% =====================================================================
    function runLoop()
        tprev = tic;
        while isvalid(ui.fig)
            dtReal = toc(tprev); tprev = tic;
            if S.playing
                dt = min(dtReal,0.05) * S.speed * 3;     % sim-hours this frame
                S.hour = mod(S.hour + dt, 24);
            else
                dt = 0;
            end
            k = floor(mod(S.hour,24)) + 1;

            % ---- sensors (live or modeled) ----
            ghi  = interpDay(S.day.ghi,  S.hour);
            temp = interpDay(S.day.temp, S.hour);
            ppt  = interpDay(S.day.ppt,  S.hour);
            load_kw  = S.meanLoad * interpShape(S.hour);
            solar_kw = (ghi/1000) * S.kwp * P.pv_derating;

            % ---- LSTM forecast on the last-24h window ----
            win = buildWindow(k);
            [solar_fc, load_fc] = lstmForward(win);

            % ---- residual-PPO decision ----
            baseAct = max(min((solar_kw - load_kw)/S.maxP, 1), -1);
            obs  = buildObs(solar_fc, load_fc, S.hour, S.day.month, baseAct);
            a    = ppoAction(obs);
            act  = max(min(baseAct + P.residual_scale*a, 1), -1);
            power = act * S.maxP;                         % + charge, - discharge

            % ---- battery + plant ----
            info = batteryStep(power, solar_kw, load_kw, max(dt,1e-3));

            % ---- render ----
            updateScene(ghi, solar_kw, load_kw, info);
            updateHUD(ghi, solar_kw, load_kw, solar_fc, load_fc, baseAct, a, act, info);
            alarm(info.unmet > 0.5);
            drawnow limitrate;
        end
    end

% =====================================================================
% REAL NETWORKS (matrix math)
% =====================================================================
    function [solar_fc, load_fc] = lstmForward(win)
        Xs = win .* P.sx_scale + P.sx_min;          % 24x7 MinMax-scaled
        H = P.hidden; h0=zeros(H,1); c0=zeros(H,1); h1=zeros(H,1); c1=zeros(H,1);
        for t = 1:size(Xs,1)
            x  = Xs(t,:).';
            g0 = P.lstm_Wih0*x  + P.lstm_bih0.' + P.lstm_Whh0*h0 + P.lstm_bhh0.';
            [h0,c0] = cell_(g0,c0,H);
            g1 = P.lstm_Wih1*h0 + P.lstm_bih1.' + P.lstm_Whh1*h1 + P.lstm_bhh1.';
            [h1,c1] = cell_(g1,c1,H);
        end
        out = P.fc_W*h1 + P.fc_b.';                 % 96x1
        fc  = reshape(out,2,48).';                  % 48x2 [solar load], normalised
        solar_fc = min(max(fc(:,1),0),1);
        load_fc  = min(max(fc(:,2),0),1);
    end
    function [h,c] = cell_(g,cprev,H)
        i=sig(g(1:H)); f=sig(g(H+1:2*H)); gg=tanh(g(2*H+1:3*H)); o=sig(g(3*H+1:4*H));
        c = f.*cprev + i.*gg;  h = o.*tanh(c);
    end

    function a = ppoAction(obsRaw)
        on = (obsRaw - P.vn_mean.') ./ sqrt(P.vn_var.' + P.vn_eps);
        on = min(max(on, -P.vn_clip), P.vn_clip);
        h  = tanh(P.pi_W0*on + P.pi_b0.');
        h  = tanh(P.pi_W1*h  + P.pi_b1.');
        a  = P.act_W*h + P.act_b.';
        a  = min(max(a,-1),1);
    end

    function obs = buildObs(solar_fc, load_fc, hour, month, baseAct)
        h2 = P.obs_hourly;
        d2s = mean(solar_fc(h2+1:end)); d2l = mean(load_fc(h2+1:end));
        obs = [ S.soc; S.soh;
                sin(2*pi*hour/24);  cos(2*pi*hour/24);
                sin(2*pi*month/12); cos(2*pi*month/12);
                solar_fc(1:h2); load_fc(1:h2);
                d2s; d2l;
                S.kwp/P.max_solar_kwp; S.kwh/P.max_bat_kwh; S.meanLoad/P.max_load_kw;
                baseAct ];
    end

    function info = batteryStep(power, solar_kw, load_kw, dt)
        eff_cap = S.kwh*S.soh; stored = S.soc*eff_cap;
        if power >= 0
            maxch = (P.soc_max-S.soc)*eff_cap / P.eta / dt;  bp = min(power, maxch);
        else
            maxdis = (S.soc-P.soc_min)*eff_cap * P.eta / dt; bp = max(power, -maxdis);
        end
        supply = solar_kw + max(0,-bp);  demand = load_kw + max(0,bp);
        if supply >= demand, unmet = 0; else, unmet = min(demand-supply, load_kw); end
        ac = max(0,bp); ad = max(0,-bp);
        S.soc = min(max((stored + (ac*P.eta - ad/P.eta)*dt)/eff_cap, P.soc_min), P.soc_max);
        S.soh = max(0.80, S.soh - (ac+ad)*dt*2e-7);
        info.bp = bp; info.unmet = unmet; info.served = load_kw - unmet;
    end

% =====================================================================
% WEATHER (sensors)  -- live via Open-Meteo, modeled fallback
% =====================================================================
    function fetchWeather(lat, lon, name)
        setStatus(sprintf('Fetching weather for %s ...', name));
        try
            o = weboptions('Timeout', 15);
            d = webread("https://api.open-meteo.com/v1/forecast", ...
                'latitude',lat, 'longitude',lon, ...
                'hourly','shortwave_radiation,temperature_2m,precipitation', ...
                'past_days',1, 'forecast_days',1, 'timezone','auto', o);
            h = d.hourly;
            ghi = h.shortwave_radiation(1:24); temp = h.temperature_2m(1:24);
            ppt = h.precipitation(1:24);
            t0  = datetime(h.time{1}, 'InputFormat','yyyy-MM-dd''T''HH:mm');
            S.day.ghi = ghi(:).'; S.day.temp = temp(:).'; S.day.ppt = ppt(:).';
            S.day.month = month(t0); S.day.dow = mod(weekday(t0)-2,7);
            S.day.date = datestr(t0,'yyyy-mm-dd');
            S.dataMsg = sprintf('LIVE weather · %s · %s', name, S.day.date);
            setStatus(S.dataMsg);
            S.soc = 0.55; S.soh = 1.0;
            ui.loc.Text = sprintf('%s   %.2f, %.2f   measured %s', name, lat, lon, S.day.date);
        catch ME
            S.day = synthDay(S); S.dataMsg = 'live feed unavailable — modeled day';
            setStatus(S.dataMsg);
            ui.loc.Text = sprintf('%s   %.2f, %.2f   (modeled)', name, lat, lon);
        end
    end

    function d = synthDay(S0) %#ok<INUSD>
        t = 0:23;
        day = max(0, sin(pi*(t-6)/12));
        d.ghi  = 1050 * day.^1.15 .* (0.8 + 0.2*sin(0.7*t));
        d.ghi  = max(0, d.ghi);
        d.temp = 26 + 6*sin(pi*(t-9)/12);
        d.ppt  = zeros(1,24);
        d.month = 6; d.dow = 3; d.date = 'modeled';
    end

    function v = interpDay(arr, h)
        i = floor(mod(h,24)) + 1; j = mod(i,24) + 1; f = mod(h,24) - floor(mod(h,24));
        v = arr(i)*(1-f) + arr(j)*f;
    end
    function v = interpShape(h)
        i = floor(mod(h,24)) + 1; j = mod(i,24) + 1; f = mod(h,24) - floor(mod(h,24));
        v = S.LOAD_SHAPE(i)*(1-f) + S.LOAD_SHAPE(j)*f;
    end

    function win = buildWindow(k)
        % last 24 hourly rows ending at k:  [ssrd tp temp load hour month dow]
        win = zeros(24,7);
        for n = 1:24
            idx = mod(k-24+n-1, 24) + 1; hr = idx-1;
            win(n,:) = [ S.day.ghi(idx), S.day.ppt(idx), S.day.temp(idx), ...
                         S.meanLoad*S.LOAD_SHAPE(idx), hr, S.day.month, S.day.dow ];
        end
    end

% =====================================================================
% UI + 3D SCENE
% =====================================================================
    function buildUI()
        ui.fig = uifigure('Name','SREP Mini-Grid — 3D Digital Twin (real LSTM+PPO)', ...
            'Color',[0.04 0.06 0.09], 'Position',[60 60 1280 760]);
        g = uigridlayout(ui.fig,[2 2],'RowHeight',{'1x',76},'ColumnWidth',{'1x',330}, ...
            'BackgroundColor',[0.04 0.06 0.09],'Padding',[10 10 10 10],'RowSpacing',8,'ColumnSpacing',8);

        % ---- 3D axes ----
        ax = uiaxes(g); ax.Layout.Row=1; ax.Layout.Column=1; ui.ax=ax;
        ax.Color=[0.06 0.10 0.14]; ax.XColor='none'; ax.YColor='none'; ax.ZColor='none';
        hold(ax,'on'); axis(ax,'equal'); ax.Clipping='off';
        view(ax,-37,22); camva(ax,8); ax.XLim=[-2 42]; ax.YLim=[-2 32]; ax.ZLim=[0 14];
        light(ax,'Position',[20 -20 40],'Style','infinite'); lighting(ax,'gouraud'); material(ax,'dull');
        buildScene(ax);

        % ---- HUD ----
        hud = uigridlayout(g,[12 1],'BackgroundColor',[0.07 0.12 0.16], ...
            'RowHeight',{26,22,22,22,22,10,22,22,10,44,10,'1x'},'Padding',[14 12 14 12],'RowSpacing',5);
        hud.Layout.Row=1; hud.Layout.Column=2;
        ui.clock = lbl(hud,'06:00',20,'w','b');
        ui.loc   = lbl(hud,'—',11,[0.55 0.70 0.74],'n');
        ui.ghi   = lbl(hud,'Irradiance (sensor): 0 W/m²',12,'w','n');
        ui.solar = lbl(hud,'Solar: 0 kW',12,[1 0.69 0.18],'n');
        ui.load  = lbl(hud,'Load: 0 kW',12,[1 0.44 0.38],'n');
        lbl(hud,'',8,'w','n');
        ui.soc   = lbl(hud,'Battery SoC: 55%',12,[0.15 0.83 0.77],'n');
        ui.soh   = lbl(hud,'Battery SoH: 100.0%',12,[0.42 0.84 0.6],'n');
        lbl(hud,'',8,'w','n');
        ui.dec   = lbl(hud,'PPO: initialising',14,'w','b'); ui.dec.WordWrap='on';
        lbl(hud,'',8,'w','n');
        ui.stat  = lbl(hud,'Powered',13,[0.22 0.85 0.54],'b');

        % ---- controls ----
        c = uigridlayout(g,[1 12],'BackgroundColor',[0.07 0.12 0.16], ...
            'ColumnWidth',{70,'1x',70,70,80,90,70,70,60,'1x',60,60}, ...
            'Padding',[12 12 12 12],'ColumnSpacing',6); c.Layout.Row=2; c.Layout.Column=[1 2];
        ui.play = uibutton(c,'Text','Pause','ButtonPushedFcn',@(b,~)onPlay(b));
        uilabel(c,'Text','');
        uilabel(c,'Text','Lat','FontColor','w');  ui.lat = uieditfield(c,'numeric','Value',6.30);
        uilabel(c,'Text','Lon','FontColor','w');  ui.lon = uieditfield(c,'numeric','Value',0.05);
        ui.town = uidropdown(c,'Items',{'Akosombo','Tamale','Kumasi','Axim','Bolgatanga','Accra'}, ...
            'ValueChangedFcn',@(d,~)onTown(d));
        ui.loadb = uibutton(c,'Text','Load 24h','BackgroundColor',[0.35 0.82 1], ...
            'ButtonPushedFcn',@(~,~)onLoad());
        ui.tier = uidropdown(c,'Items',{'50 kWp','75 kWp','120 kWp'},'Value','75 kWp', ...
            'ValueChangedFcn',@(d,~)onTier(d));
        uilabel(c,'Text','');
        uilabel(c,'Text','');
        ui.mute = uibutton(c,'Text','Alarm','ButtonPushedFcn',@(b,~)onMute(b));
        uilabel(c,'Text','');
    end

    function buildScene(ax)
        % ground
        patch(ax,'XData',[-2 42 42 -2],'YData',[-2 -2 32 32],'ZData',[0 0 0 0], ...
              'FaceColor',[0.30 0.33 0.22],'EdgeColor','none');
        % solar array (tilted panels)
        for r=0:2, for cc=0:3
            px=4+cc*3; py=16+r*3;
            V=[px-1.2 py-0.9 1.6; px+1.2 py-0.9 1.6; px+1.2 py+0.9 2.6; px-1.2 py+0.9 2.6];
            patch(ax,'Vertices',V,'Faces',[1 2 3 4],'FaceColor',[0.10 0.22 0.42], ...
                  'EdgeColor',[0.2 0.4 0.7],'FaceAlpha',1,'SpecularStrength',0.9);
            line(ax,[px px],[py-0.9 py+0.9],[0 0],'Color',[0.4 0.4 0.4]); % support shadow hint
        end, end
        % sensor mast
        line(ax,[2 2],[26 26],[0 5],'Color',[0.7 0.75 0.8],'LineWidth',3);
        ui.pyra = scatter3(ax,2.8,26,4.6,60,[1 0.7 0.2],'filled');           % pyranometer
        scatter3(ax,2,26,5.1,40,[0.85 0.9 0.95],'filled');                   % anemometer
        text(ax,2,26,6.0,'Weather sensors','Color',[0.8 0.9 0.92],'FontSize',9,'HorizontalAlignment','center');
        % battery + SoC fill
        ui.battBox = cuboid(ax,[20 8 0],[7 3 3],[0.18 0.24 0.28],1);
        ui.socFill = cuboid(ax,[20 8 0.1],[6.4 2.6 1.5],[0.15 0.83 0.77],0.95);
        text(ax,20,8,3.6,'Battery storage','Color',[0.8 0.9 0.92],'FontSize',9,'HorizontalAlignment','center');
        % AI EMS rack
        cuboid(ax,[26 13 0],[2.2 1.6 3],[0.07 0.10 0.13],1);
        ui.emsLed = scatter3(ax,26,13.9,2.6,50,[0.35 0.82 1],'filled');
        text(ax,26,13,4.0,'AI EMS (LSTM+PPO)','Color',[0.35 0.82 1],'FontSize',9,'HorizontalAlignment','center');
        % inverter
        cuboid(ax,[24 18 0],[3 3 2.6],[0.7 0.7 0.62],1);
        text(ax,24,18,3.4,'Inverter / PCS','Color',[0.8 0.9 0.92],'FontSize',8,'HorizontalAlignment','center');
        % houses
        ui.win = gobjects(1,4); hp=[34 22; 37 25; 33 26; 38 21];
        for n=1:4
            cuboid(ax,[hp(n,1) hp(n,2) 0],[2.2 2.2 2.2],[0.78 0.72 0.6],1);
            ui.win(n)=scatter3(ax,hp(n,1),hp(n,2)-1.15,1.2,40,[0.2 0.24 0.28],'filled','MarkerFaceAlpha',1);
        end
        text(ax,35.5,23,3.4,'Village load','Color',[1 0.44 0.38],'FontSize',9,'HorizontalAlignment','center');
        % power lines
        line(ax,[8 24],[18 18],[2.4 2.4],'Color',[0.25 0.28 0.3],'LineWidth',1.5);
        line(ax,[24 35.5],[18 23],[2.4 2.4],'Color',[0.25 0.28 0.3],'LineWidth',1.5);
        line(ax,[24 20],[16 9],[2.4 2.4],'Color',[0.25 0.28 0.3],'LineWidth',1.5);
        % sun marker + flow markers
        ui.sun  = scatter3(ax,0,0,0,260,[1 0.85 0.4],'filled');
        ui.fPV  = scatter3(ax,nan,nan,nan,40,[1 0.69 0.18],'filled');
        ui.fLD  = scatter3(ax,nan,nan,nan,40,[1 0.44 0.38],'filled');
        ui.fBT  = scatter3(ax,nan,nan,nan,40,[0.15 0.83 0.77],'filled');
        ui.phase = 0;
    end

    function updateScene(ghi, solar_kw, load_kw, info)
        ax = ui.ax;
        % sun position by hour
        ang = (S.hour/24)*2*pi - pi/2;
        ui.sun.XData = 20 + cos(ang)*30; ui.sun.YData = -10;
        ui.sun.ZData = max(0.5, sin(ang)*22);
        ui.sun.MarkerFaceColor = (sin(ang)>0).*[1 0.85 0.4] + (sin(ang)<=0).*[0.4 0.45 0.6];
        % pyranometer glow ~ irradiance
        gl = min(1, ghi/1000); ui.pyra.SizeData = 40 + 120*gl;
        % SoC fill height
        zf = max(0.05, 2.8*S.soc);
        c=[20 8]; w=[3.2 1.3];
        V=[c(1)-w(1) c(2)-w(2) 0.1; c(1)+w(1) c(2)-w(2) 0.1; c(1)+w(1) c(2)+w(2) 0.1; c(1)-w(1) c(2)+w(2) 0.1; ...
           c(1)-w(1) c(2)-w(2) 0.1+zf; c(1)+w(1) c(2)-w(2) 0.1+zf; c(1)+w(1) c(2)+w(2) 0.1+zf; c(1)-w(1) c(2)+w(2) 0.1+zf];
        ui.socFill.Vertices = V;
        col = [0.15 0.83 0.77]; if S.soc<0.2, col=[1 0.33 0.43]; elseif S.soc<0.4, col=[1 0.70 0.29]; end
        ui.socFill.FaceColor = col;
        % EMS led pulse
        ui.emsLed.SizeData = 40 + 40*abs(sin(S.hour*3));
        % houses powered / unmet
        if info.unmet>0.5, wc=[1 0.33 0.43]; else, wc=[1 0.81 0.43]; end
        if interpDay(S.day.ghi,S.hour)<5 || info.unmet>0.5
            for n=1:4, ui.win(n).MarkerFaceColor = wc; ui.win(n).SizeData=70; end
        else
            for n=1:4, ui.win(n).MarkerFaceColor = [0.2 0.24 0.28]; ui.win(n).SizeData=40; end
        end
        % flow markers
        ui.phase = mod(ui.phase + 0.05*(1+solar_kw/30), 1);
        seg = @(A,B,t) A + (B-A)*t;
        if solar_kw>0.5
            p = seg([8 18 2.6],[24 18 2.6],ui.phase); set(ui.fPV,'XData',p(1),'YData',p(2),'ZData',p(3));
        else, set(ui.fPV,'XData',nan); end
        if info.served>0.5
            p = seg([24 18 2.6],[35.5 23 2.6],ui.phase); set(ui.fLD,'XData',p(1),'YData',p(2),'ZData',p(3));
        else, set(ui.fLD,'XData',nan); end
        if abs(info.bp)>0.5
            if info.bp>0, p = seg([24 16 2.6],[20 9 2.6],ui.phase);   % charge -> battery
            else,         p = seg([20 9 2.6],[24 16 2.6],ui.phase); end % discharge -> bus
            set(ui.fBT,'XData',p(1),'YData',p(2),'ZData',p(3));
        else, set(ui.fBT,'XData',nan); end
    end

    function updateHUD(ghi, solar_kw, load_kw, sfc, lfc, baseAct, a, act, info)
        ui.clock.Text = sprintf('%02d:%02d', floor(S.hour), floor(mod(S.hour,1)*60));
        ui.ghi.Text   = sprintf('Irradiance (sensor): %d W/m²', round(ghi));
        ui.solar.Text = sprintf('Solar: %d kW', round(solar_kw));
        ui.load.Text  = sprintf('Load: %d kW', round(load_kw));
        ui.soc.Text   = sprintf('Battery SoC: %d%%', round(S.soc*100));
        ui.soh.Text   = sprintf('Battery SoH: %.1f%%', S.soh*100);
        if info.bp > 0.5
            d = sprintf('PPO: charge %d kW   (baseline %+.2f, residual %+.2f)', round(info.bp), baseAct, P.residual_scale*a);
        elseif info.bp < -0.5
            d = sprintf('PPO: discharge %d kW   (baseline %+.2f, residual %+.2f)', round(-info.bp), baseAct, P.residual_scale*a);
        else
            d = sprintf('PPO: holding   (baseline %+.2f, residual %+.2f)', baseAct, P.residual_scale*a);
        end
        ui.dec.Text = d;
        if info.unmet < 0.5
            ui.stat.Text = 'Powered — load fully served'; ui.stat.FontColor=[0.22 0.85 0.54];
        else
            ui.stat.Text = sprintf('SHORTFALL — %d kW unmet', round(info.unmet)); ui.stat.FontColor=[1 0.33 0.43];
        end
    end

    function alarm(on)
        if on && ~S.muted                          % throttled beep (~every 0.6 s)
            tsec = posixtime(datetime('now'));
            if tsec - S.lastBeep > 0.6
                Fs=8000; tt=0:1/Fs:0.15; y=0.3*sin(2*pi*880*tt);
                try, sound(y,Fs); catch, end
                S.lastBeep = tsec;
            end
        end
        if on, ui.fig.Color=[0.18 0.04 0.07]; else, ui.fig.Color=[0.04 0.06 0.09]; end
    end

% =====================================================================
% callbacks + helpers
% =====================================================================
    function onPlay(b), S.playing=~S.playing; b.Text=ternary(S.playing,'Pause','Play'); end
    function onMute(b), S.muted=~S.muted; b.Text=ternary(S.muted,'Muted','Alarm'); end
    function onTier(d)
        S.tierIdx = find(strcmp(d.Value,{'50 kWp','75 kWp','120 kWp'})); S=applyTier(S);
    end
    function onTown(d)
        coords=[6.30 0.05;9.40 -0.84;6.69 -1.62;4.87 -2.24;10.79 -0.85;5.56 -0.20];
        i=find(strcmp(d.Value,{'Akosombo','Tamale','Kumasi','Axim','Bolgatanga','Accra'}));
        ui.lat.Value=coords(i,1); ui.lon.Value=coords(i,2);
    end
    function onLoad(), fetchWeather(ui.lat.Value, ui.lon.Value, ui.town.Value); end
    function setStatus(t), ui.fig.Name = sprintf('SREP 3D Twin  —  %s', t); end

    function Sx = applyTier(Sx)
        T=[50 160 5.6;75 237 8.4;120 378 13.4];
        Sx.kwp=T(Sx.tierIdx,1); Sx.kwh=T(Sx.tierIdx,2); Sx.meanLoad=T(Sx.tierIdx,3);
        Sx.maxP=min(0.5*Sx.kwh, 0.8*Sx.kwp);
    end

    function h = cuboid(ax,cen,sz,col,al)
        x=cen(1); y=cen(2); z=cen(3); wx=sz(1)/2; wy=sz(2)/2; wz=sz(3);
        V=[x-wx y-wy z; x+wx y-wy z; x+wx y+wy z; x-wx y+wy z; ...
           x-wx y-wy z+wz; x+wx y-wy z+wz; x+wx y+wy z+wz; x-wx y+wy z+wz];
        F=[1 2 3 4;5 6 7 8;1 2 6 5;2 3 7 6;3 4 8 7;4 1 5 8];
        h=patch(ax,'Vertices',V,'Faces',F,'FaceColor',col,'EdgeColor',[0 0 0], ...
                'EdgeAlpha',0.2,'FaceAlpha',al);
    end

    function L = lbl(parent,txt,sz,col,w)
        L=uilabel(parent,'Text',txt,'FontSize',sz,'FontColor',col);
        if strcmp(w,'b'), L.FontWeight='bold'; end
    end
    function v = ternary(t,a,b), if t, v=a; else, v=b; end, end

    function P = loadPack()
        f = 'srep_matlab_pack.mat';
        if ~isfile(f), f = fullfile('matlab_export','srep_matlab_pack.mat'); end
        assert(isfile(f), ['Cannot find srep_matlab_pack.mat. Run ' ...
            'dump_weights_for_matlab.py first and put the file next to this script.']);
        P = load(f);
    end
end
