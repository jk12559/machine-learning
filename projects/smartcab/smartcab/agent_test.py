# -*- coding: utf-8 -*-
import unittest
import mock
import random

from agent import LearningAgent


class test_agent(unittest.TestCase):
    
    @mock.patch('agent.RoutePlanner', autospec=True)
    @mock.patch('agent.Environment', autospec=True)
    def setUp(self, mock_env, mock_planner):
        mock_env.valid_actions = [None, 'forward', 'left', 'right']
        self.agent = LearningAgent(mock_env)
        self.agent.planner = mock_planner
        self.state = ('forward', 'red', None, 'left', 'right')
    
    def test_reset(self):
        pass
    
    def test_build_state(self):
        '''verifies that state values are correctly extracted from the agent and formatted to the tuple'''
        self.agent.planner.next_waypoint.return_value = 'forward'
        self.agent.env.sense.return_value = {'light': 'red',
                                   'left': None,
                                   'oncoming': 'left',
                                   'right': 'right'}
        self.assertEqual(self.agent.build_state(), self.state)
    
    def test_get_maxQ(self):
        '''
        Test cases:
            Case 1: Q[state] = {None: 0, 'left':1, 'forward':2,'right':3}, maxQ == 3
            Case 2: Q[state] = {None: 0, 'left':0, 'forward':0,'right':0}, maxQ == 0
        '''
        
        #Case 1
        self.agent.Q[self.state] = {None: 0, 'left':1, 'forward':2,'right':3}
        maxQ = self.agent.get_maxQ(self.state)
        self.assertEquals(maxQ, 3, msg = 'Case 1: expected 3 but got {}'.format(maxQ))
        
        #Case 2
        self.agent.Q[self.state] = {None: 0, 'left':0, 'forward':0,'right':0}
        maxQ = self.agent.get_maxQ(self.state)
        self.assertEquals(maxQ, 0, msg = 'Case 2: expected 0 but got {}'.format(maxQ))       
        
    
    def test_createQ(self):
        '''verify that the Q table initializes empty and adds a state with initial values of 0 for each action'''
        self.assertDictEqual(self.agent.Q, {})
        self.agent.learning = False
        self.agent.createQ(self.state)
        self.assertNotIn(self.state, self.agent.Q, msg="state added when learning set to False")
        
        self.agent.learning = True
        self.agent.createQ(self.state)
        self.assertIn(self.state, self.agent.Q, msg="state not added when learning set to True")
        
        self.assertEqual(set(self.agent.valid_actions), set(self.agent.Q[self.state].keys()), msg="valid actions not added to Q-list properly")
        
        for action, q_value in self.agent.Q[self.state].iteritems():
            self.assertEqual(q_value, 0.0, msg="{} Q-value not properly initialized to 0.0".format(action))
    
    @mock.patch('random.choice') 
    @mock.patch('random.random')
    @mock.patch('agent.LearningAgent.get_maxQ')
    def test_choose_action(self, mock_maxQ, mock_random, mock_choice):
        '''
        Test cases:
            case 1: if not learning, random.choice gets called
            case 2: if learning, state = ('forward', 'red', None, 'left', 'right'), and Q[state] = {None: 0, 'left':1, 'forward':2,'right':3}, action == 'right'
            case 3: if learning, state = ('forward', 'red', None, 'left', 'right'), and Q[state] = {None: 3, 'left':2, 'forward':1,'right':0}, action == None
            case 4: if learning, state = ('forward', 'red', None, 'left', 'right'), and Q[state] = {None: 0, 'left':0, 'forward':2,'right':0}, action == 'forward'
            case 5: if learning, state = ('forward', 'red', None, 'left', 'right'), and Q[state] = {None: 0, 'left':0, 'forward':0,'right':0}, random.choice is called
            case 6: if learning, state = ('forward', 'red', None, 'left', 'right'), ,random < epsilon, and Q[state] = {None: 0, 'left':1, 'forward':2,'right':3}, random.choice gets called
        '''
        mock_random.return_value = self.agent.epsilon + .1
        #case 1
        self.agent.learning = False
        self.agent.choose_action(self.state)
        self.assertTrue(mock_choice.called, msg='case 1: random choice not made when learning is false')
        
        #case 2
        self.agent.learning = True
        mock_maxQ.return_value = 3
        self.agent.Q[self.state] = {None: 0, 'left':1, 'forward':2,'right':3}
        action = self.agent.choose_action(self.state)
        self.assertEqual(action, 'right', msg='case 2: expected "right" but got {}'.format(action))
        
        #case 3
        self.agent.learning = True
        mock_maxQ.return_value = 3
        self.agent.Q[self.state] = {None: 3, 'left':2, 'forward':1,'right':0}
        action = self.agent.choose_action(self.state)
        self.assertEqual(action, None, msg='case 3: expected None but got {}'.format(action))
        
        #case 4
        self.agent.learning = True
        mock_maxQ.return_value = 2
        self.agent.Q[self.state] = {None: 0, 'left':0, 'forward':2,'right':0}
        action = self.agent.choose_action(self.state)
        self.assertEqual(action, 'forward', msg='case 4: expected "forward" but got {}'.format(action))
        
        #case 5
        self.agent.learning = True
        mock_maxQ.return_value = 0
        self.agent.Q[self.state] = {None: 0, 'left':0, 'forward':0,'right':0}
        action = self.agent.choose_action(self.state)
        self.assertTrue(mock_choice.called, msg='case 5: random choice not made when multiple actions have the same maxQ value')
        
        #case 6
        mock_random.return_value = self.agent.epsilon - .1
        self.agent.learning = True
        mock_maxQ.return_value = 3
        self.agent.Q[self.state] = {None: 0, 'left':1, 'forward':2,'right':3}
        action = self.agent.choose_action(self.state)
        self.assertTrue(mock_choice.called, msg='case 6: random choice not made when random value is less than epsilon')
    
    def test_learn(self):
        '''
        Test cases:
            Case 1: alpha = .1, Q[state] =Q[state] = {None: 0, 'left':1, 'forward':2,'right':3}, action = 'forward', reward = 5 => Q[state] = {None: 0, 'left':1, 'forward':2.3,'right':3}               
            Case 2: alpha = .1, Q[state] =Q[state] = {None: 0, 'left':1, 'forward':2,'right':3}, action = 'forward', reward = -5 => Q[state] = {None: 0, 'left':1, 'forward':1.3,'right':3}               
            Case 3: alpha = .1, Q[state] =Q[state] = {None: 0, 'left':1, 'forward':2,'right':3}, action = None, reward = 5 => Q[state] = {None: .5, 'left':1, 'forward':2,'right':3}               
            Case 4: alpha = .5, Q[state] =Q[state] = {None: 0, 'left':1, 'forward':2,'right':3}, action = 'forward', reward = 5 => Q[state] = {None: 0, 'left':1, 'forward':4.3,'right':3}   
        '''
        self.agent.learning = True
        #case 1
        self.agent.alpha = .1
        self.agent.Q[self.state] = {None: 0, 'left':1, 'forward':2,'right':3}
        self.agent.learn(self.state, 'forward', 5)
        self.assertDictEqual(self.agent.Q[self.state], {None: 0, 'left':1, 'forward':2.3,'right':3}, msg="case 1: Expected {} but got {}".format({None: 0, 'left':1, 'forward':2.3,'right':3}, self.agent.Q[self.state]))
        #case 2
        self.agent.alpha = .1
        self.agent.Q[self.state] = {None: 0, 'left':1, 'forward':2,'right':3}
        self.agent.learn(self.state, 'forward', -5)
        self.assertDictEqual(self.agent.Q[self.state], {None: 0, 'left':1, 'forward':1.3,'right':3}, msg="case 2: Expected {} but got {}".format({None: 0, 'left':1, 'forward':1.3,'right':3},self.agent.Q[self.state]))
        #case 3
        self.agent.alpha = .1
        self.agent.Q[self.state] = {None: 0, 'left':1, 'forward':2,'right':3}
        self.agent.learn(self.state, None, 5)
        self.assertDictEqual(self.agent.Q[self.state], {None: .5, 'left':1, 'forward':2,'right':3}, msg="case 3: Expected {} but got {}".format({None: .5, 'left':1, 'forward':2,'right':3},self.agent.Q[self.state]))
        #case 4
        self.agent.alpha = .5
        self.agent.Q[self.state] = {None: 0, 'left':1, 'forward':2,'right':3}
        self.agent.learn(self.state, 'forward', 5)
        self.assertDictEqual(self.agent.Q[self.state], {None: 0, 'left':1, 'forward':3.5,'right':3}, msg="case 4: Expected {} but got {}".format({None: 0, 'left':1, 'forward':3.5,'right':3},self.agent.Q[self.state]))
    
    
    
    
unittest.main()