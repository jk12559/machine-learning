# -*- coding: utf-8 -*-
import unittest
import mock

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
        pass
    
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
        
    def test_choose_action(self):
        pass
    
    def test_learn(self):
        pass
    
    def test_update(self):
        pass
    
    
    
unittest.main()